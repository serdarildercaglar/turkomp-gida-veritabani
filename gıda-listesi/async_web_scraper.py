import pandas as pd
import json
import time
import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Scraping yapılandırma parametreleri"""
    max_concurrent_requests: int = 10  # Eş zamanlı istek sayısı
    rate_limit_delay: float = 0.5      # İstekler arası minimum süre (saniye)
    request_timeout: int = 30          # İstek timeout süresi
    max_retries: int = 3               # Maksimum tekrar deneme sayısı
    retry_delay: float = 2.0           # Tekrar deneme arası bekleme
    checkpoint_interval: int = 50      # Her N işlemde checkpoint kaydet
    
@dataclass
class ScrapingResult:
    """Scraping sonuç veri yapısı"""
    success: bool
    data: Optional[List[Dict]]
    error: Optional[str]
    url: str
    attempts: int

class ProgressManager:
    """İlerleme durumu ve checkpoint yönetimi"""
    
    def __init__(self, checkpoint_file: str = "scraping_progress.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.failed_urls = []
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()
        
    def load_checkpoint(self) -> Dict:
        """Checkpoint dosyasından ilerleme durumunu yükle"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    logger.info(f"Checkpoint yüklendi: {checkpoint.get('processed_count', 0)} işlem tamamlanmış")
                    return checkpoint
            except Exception as e:
                logger.error(f"Checkpoint yüklenirken hata: {e}")
        return {}
    
    def save_checkpoint(self, df: pd.DataFrame, current_index: int):
        """Mevcut durumu checkpoint dosyasına kaydet"""
        checkpoint_data = {
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'current_index': current_index,
            'failed_urls': self.failed_urls,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint kaydedildi: {self.processed_count} işlem")
        except Exception as e:
            logger.error(f"Checkpoint kaydedilirken hata: {e}")
    
    def update_stats(self, result: ScrapingResult):
        """İstatistikleri güncelle"""
        self.processed_count += 1
        if result.success:
            self.success_count += 1
        else:
            self.error_count += 1
            self.failed_urls.append({
                'url': result.url,
                'error': result.error,
                'attempts': result.attempts,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_progress_info(self, total_count: int) -> str:
        """İlerleme bilgisi döndür"""
        elapsed = time.time() - self.start_time
        if self.processed_count > 0:
            avg_time = elapsed / self.processed_count
            remaining = (total_count - self.processed_count) * avg_time
            eta = datetime.fromtimestamp(time.time() + remaining).strftime('%H:%M:%S')
        else:
            eta = "Hesaplanıyor..."
        
        return (f"İşlenen: {self.processed_count}/{total_count} "
                f"Başarılı: {self.success_count} "
                f"Hatalı: {self.error_count} "
                f"ETA: {eta}")

class AsyncWebScraper:
    """Asenkron web scraper sınıfı"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.rate_limiter = asyncio.Semaphore(1)  # Rate limiting için
        self.last_request_time = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests * 2,
            limit_per_host=self.config.max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Rate limiting uygula"""
        async with self.rate_limiter:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.config.rate_limit_delay:
                await asyncio.sleep(self.config.rate_limit_delay - time_since_last)
            self.last_request_time = time.time()
    
    async def _fetch_with_retry(self, url: str) -> ScrapingResult:
        """Retry mekanizması ile HTTP isteği gönder"""
        last_error = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                await self._rate_limit()
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        data = self._extract_data_from_html(content, url)
                        return ScrapingResult(
                            success=True,
                            data=data,
                            error=None,
                            url=url,
                            attempts=attempt
                        )
                    else:
                        last_error = f"HTTP {response.status}"
                        
            except asyncio.TimeoutError:
                last_error = "Timeout"
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(f"Attempt {attempt} failed for {url}: {last_error}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        return ScrapingResult(
            success=False,
            data=None,
            error=last_error,
            url=url,
            attempts=self.config.max_retries
        )
    
    def _extract_data_from_html(self, html_content: str, url: str) -> Optional[List[Dict]]:
        """HTML içeriğinden veri çıkar"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table', {'id': 'foodResultlist'})
            
            if not table:
                logger.warning(f"Table bulunamadı: {url}")
                return None
            
            data = []
            for row in table.select('tbody tr'):
                cols = row.find_all('td')
                if len(cols) >= 5:
                    bileşen = cols[0].get_text(strip=True)
                    birim = cols[1].get_text(strip=True)
                    ort = cols[2].get_text(strip=True).replace(',', '.')
                    min_ = cols[3].get_text(strip=True).replace(',', '.')
                    max_ = cols[4].get_text(strip=True).replace(',', '.')
                    
                    def to_float(val):
                        try:
                            return float(val)
                        except:
                            return val
                    
                    data.append({
                        'Bileşen': bileşen,
                        'Birim': birim,
                        'Ortalama': to_float(ort),
                        'Minimum': to_float(min_),
                        'Maksimum': to_float(max_)
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"HTML parsing hatası {url}: {str(e)}")
            return None
    
    async def scrape_url(self, url: str) -> ScrapingResult:
        """Tek URL için scraping işlemi"""
        async with self.semaphore:
            return await self._fetch_with_retry(url)

class GidaDataProcessor:
    """Ana gıda veri işleme sınıfı"""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.progress = ProgressManager()
        
    def _prepare_dataframe(self, df: pd.DataFrame, checkpoint: Dict) -> Tuple[pd.DataFrame, int]:
        """DataFrame'i hazırla ve başlangıç pozisyonunu belirle"""
        # Orijinal sütunlar - korunmalı
        if "Data" not in df.columns:
            df["Data"] = None
            
        # Ek sütunlar - yeni özellikler için
        if "Status" not in df.columns:
            df["Status"] = "Pending"
        if "Attempts" not in df.columns:
            df["Attempts"] = 0
        if "Last_Error" not in df.columns:
            df["Last_Error"] = None
            
        start_index = checkpoint.get('current_index', 0)
        self.progress.processed_count = checkpoint.get('processed_count', 0)
        self.progress.success_count = checkpoint.get('success_count', 0)
        self.progress.error_count = checkpoint.get('error_count', 0)
        self.progress.failed_urls = checkpoint.get('failed_urls', [])
        
        return df, start_index
    
    def _get_valid_url(self, row: pd.Series, link_columns: List[str]) -> Optional[str]:
        """Satırdan geçerli URL'yi bul"""
        for col in link_columns:
            if col in row and pd.notna(row[col]):
                url_list = str(row[col]).split(',')
                for url in url_list:
                    url = url.strip()
                    if "food" in url.lower() and url.startswith('http'):
                        return url
        return None
    
    async def process_batch(self, df: pd.DataFrame, start_idx: int, end_idx: int, 
                          link_columns: List[str]) -> List[Tuple[int, ScrapingResult]]:
        """Bir batch URL'yi işle"""
        tasks = []
        
        async with AsyncWebScraper(self.config) as scraper:
            for idx in range(start_idx, min(end_idx, len(df))):
                row = df.iloc[idx]
                
                # Daha önce başarıyla işlenmiş mi kontrol et
                if pd.notna(row.get("Data")) and row.get("Status") == "Success":
                    continue
                
                url = self._get_valid_url(row, link_columns)
                if url:
                    task = asyncio.create_task(scraper.scrape_url(url))
                    tasks.append((idx, task))
                else:
                    # URL bulunamadı - orijinal kodla aynı format
                    error_result = ScrapingResult(False, None, "Uygun URL bulunamadı", "", 0)
                    tasks.append((idx, asyncio.create_task(asyncio.coroutine(lambda: error_result)())))
            
            # Tüm task'ları bekle
            results = []
            for idx, task in tasks:
                try:
                    result = await task
                    results.append((idx, result))
                except Exception as e:
                    error_result = ScrapingResult(False, None, str(e), "", 0)
                    results.append((idx, error_result))
            
            return results
    
    async def process_all_gida(self, csv_path_in: str, csv_path_out: str):
        """Ana işleme fonksiyonu"""
        logger.info(f"Gıda verisi işleme başlatılıyor: {csv_path_in}")
        
        # DataFrame yükle
        df = pd.read_csv(csv_path_in)
        logger.info(f"Toplam {len(df)} kayıt yüklendi")
        
        # Link sütunlarını belirle
        link_columns = ["Gıda Linki", "Linkler", "Link", "URL"]
        found_columns = [col for col in link_columns if col in df.columns]
        
        if not found_columns:
            raise Exception(f"Link sütunu bulunamadı! Aranan sütunlar: {link_columns}")
        
        logger.info(f"Link sütunları bulundu: {found_columns}")
        
        # Checkpoint yükle ve DataFrame hazırla
        checkpoint = self.progress.load_checkpoint()
        df, start_index = self._prepare_dataframe(df, checkpoint)
        
        logger.info(f"İşleme {start_index}. kayıttan başlanıyor")
        
        # Batch işleme
        batch_size = self.config.max_concurrent_requests * 2
        total_batches = (len(df) - start_index + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            batch_start = start_index + (batch_num * batch_size)
            batch_end = min(batch_start + batch_size, len(df))
            
            logger.info(f"Batch {batch_num + 1}/{total_batches} işleniyor ({batch_start}-{batch_end})")
            
            try:
                # Batch'i işle
                results = await self.process_batch(df, batch_start, batch_end, found_columns)
                
                # Sonuçları DataFrame'e uygula
                for idx, result in results:
                    self.progress.update_stats(result)
                    
                    # Ek sütunları güncelle
                    df.at[idx, "Status"] = "Success" if result.success else "Failed"
                    df.at[idx, "Attempts"] = result.attempts
                    df.at[idx, "Last_Error"] = result.error if not result.success else None
                    
                    # Data sütunu - ORIJINAL FORMAT KORUNUYOR
                    if result.success and result.data:
                        # Başarılı durum: orijinal kodla aynı
                        df.at[idx, "Data"] = json.dumps(result.data, ensure_ascii=False)
                    else:
                        # Hata durumu: orijinal kodla aynı format
                        error_data = {"hata": result.error or "Bilinmeyen hata"}
                        df.at[idx, "Data"] = json.dumps(error_data, ensure_ascii=False)
                
                # Progress bilgisi göster
                progress_info = self.progress.get_progress_info(len(df))
                logger.info(progress_info)
                
                # Checkpoint kaydet
                if (batch_num + 1) % (self.config.checkpoint_interval // batch_size + 1) == 0:
                    self.progress.save_checkpoint(df, batch_end)
                    df.to_csv(csv_path_out, index=False, encoding="utf-8-sig")
                    logger.info(f"Ara kayıt tamamlandı: {csv_path_out}")
                
            except Exception as e:
                logger.error(f"Batch {batch_num + 1} işlenirken hata: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Hatalı URL'leri kaydet
        if self.progress.failed_urls:
            failed_urls_file = csv_path_out.replace('.csv', '_failed_urls.json')
            with open(failed_urls_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress.failed_urls, f, ensure_ascii=False, indent=2)
            logger.info(f"Hatalı URL'ler kaydedildi: {failed_urls_file}")
            
            # Hatalı URL'leri tekrar çekmeyi öner
            try:
                retry_choice = input(f"\n{len(self.progress.failed_urls)} hatalı URL tekrar çekilsin mi? (y/n): ").lower()
                if retry_choice == 'y':
                    logger.info("Hatalı URL'ler tekrar çekiliyor...")
                    await self._retry_failed_urls(df, csv_path_out)
            except EOFError:
                logger.info("Otomatik mod - hatalı URL'ler kaydedildi")
        
        # Final kayıt
        df.to_csv(csv_path_out, index=False, encoding="utf-8-sig")
        self.progress.save_checkpoint(df, len(df))
        
        # Özet bilgi
        total_time = time.time() - self.progress.start_time
        success_rate = (self.progress.success_count / self.progress.processed_count * 100) if self.progress.processed_count > 0 else 0
        
        logger.info(f"""
        ============ İŞLEM TAMAMLANDI ============
        Toplam işlenen: {self.progress.processed_count}
        Başarılı: {self.progress.success_count}
        Hatalı: {self.progress.error_count}
        Başarı oranı: {success_rate:.1f}%
        Toplam süre: {total_time:.1f} saniye
        Ortalama hız: {self.progress.processed_count/total_time:.1f} URL/saniye
        Çıkış dosyası: {csv_path_out}
        ==========================================
        """)
        
    async def _retry_failed_urls(self, df: pd.DataFrame, csv_path_out: str):
        """Hatalı URL'leri konservatif ayarlarla tekrar çek"""
        failed_mask = df['Status'] == 'Failed'
        failed_df = df[failed_mask].copy()
        
        if len(failed_df) == 0:
            return
            
        logger.info(f"Toplam {len(failed_df)} hatalı URL tekrar çekiliyor...")
        
        # Konservatif config
        retry_config = ScrapingConfig(
            max_concurrent_requests=2,  # Çok az eş zamanlı
            rate_limit_delay=3.0,       # Uzun bekleme
            request_timeout=60,         # Uzun timeout
            max_retries=5,              # Daha fazla deneme
            retry_delay=5.0             # Uzun retry delay
        )
        
        success_count = 0
        
        async with AsyncWebScraper(retry_config) as scraper:
            for idx, row in failed_df.iterrows():
                # Link sütunlarından URL bul
                link_columns = ["Gıda Linki", "Linkler", "Link", "URL"]
                url = self._get_valid_url(row, link_columns)
                
                if not url:
                    continue
                    
                logger.info(f"Retry: {row['Gıda']} - {url}")
                
                result = await scraper.scrape_url(url)
                
                if result.success and result.data:
                    success_count += 1
                    df.at[idx, 'Data'] = json.dumps(result.data, ensure_ascii=False)
                    df.at[idx, 'Status'] = 'Success'
                    df.at[idx, 'Attempts'] = df.at[idx, 'Attempts'] + result.attempts
                    df.at[idx, 'Last_Error'] = None
                    logger.info(f"✅ Başarılı: {row['Gıda']}")
                else:
                    # Hata durumunda da orijinal format koru
                    error_data = {"hata": result.error or "Bilinmeyen hata"}
                    df.at[idx, 'Data'] = json.dumps(error_data, ensure_ascii=False)
                    df.at[idx, 'Attempts'] = df.at[idx, 'Attempts'] + result.attempts
                    df.at[idx, 'Last_Error'] = result.error
                    logger.warning(f"❌ Hala başarısız: {row['Gıda']}")
                
                # Her 5 işlemde ara kayıt
                if success_count % 5 == 0:
                    df.to_csv(csv_path_out, index=False, encoding="utf-8-sig")
        
        logger.info(f"Retry tamamlandı. {success_count}/{len(failed_df)} URL başarılı")
        return df

# Kullanım fonksiyonu
async def main():
    """Ana çalıştırma fonksiyonu"""
    
    # Yapılandırma
    config = ScrapingConfig(
        max_concurrent_requests=20,    # Eş zamanlı istek sayısı
        rate_limit_delay=0.7,          # İstekler arası bekleme (saniye)
        request_timeout=25,            # İstek timeout (saniye)
        max_retries=5,                 # Tekrar deneme sayısı
        retry_delay=2,               # Tekrar deneme bekleme süresi
        checkpoint_interval=100        # Her 100 işlemde checkpoint
    )
    
    # İşlemci oluştur ve çalıştır
    processor = GidaDataProcessor(config)
    
    input_csv = "/home/serdar/Documents/turkomp-g-da-veritabani/turkomp_gida_listesi.csv"
    output_csv = "turkomp_gida_besinli.csv"
    
    try:
        await processor.process_all_gida(input_csv, output_csv)
    except KeyboardInterrupt:
        logger.info("İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"İşlem sırasında hata: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Event loop'u çalıştır
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nİşlem durduruldu.")
