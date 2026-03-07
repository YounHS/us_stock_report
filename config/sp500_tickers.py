"""S&P 500 종목 리스트 및 섹터 매핑"""

import pandas as pd
import requests
from io import StringIO
from typing import Dict, List

# 11 GICS Sectors
SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

# Sector ETF mapping for news fetching
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

# Market index ETFs
MARKET_ETFS = ["SPY", "QQQ", "DIA", "IWM"]

# 데이트레이딩용 인기 레버리지 ETF (Ross Cameron 전략 확장 스캔용)
LEVERAGED_ETFS = [
    # 3x 지수
    "TQQQ", "SQQQ",   # NASDAQ-100 Long/Short
    "UPRO", "SPXS",   # S&P 500 Long/Short
    "UDOW", "SDOW",   # Dow Jones Long/Short
    "TNA",  "TZA",    # Russell 2000 Long/Short
    # 3x 섹터
    "SOXL", "SOXS",   # 반도체 Long/Short
    "TECL", "TECS",   # 테크 Long/Short
    "FAS",  "FAZ",    # 금융 Long/Short
    "LABU", "LABD",   # 바이오테크 Long/Short
    "NAIL",           # 주택건설 3x Long
    "FNGU", "FNGD",   # FAANG+ 3x Long/Short
    # 2x 원자재
    "BOIL", "KOLD",   # 천연가스 Long/Short
    "UCO",  "SCO",    # 원유 Long/Short
    "NUGT", "DUST",   # 금광 Long/Short
    "JNUG", "JDST",   # 주니어 금광 Long/Short
    # 변동성
    "UVXY", "SVXY",   # VIX Long/Short
]

# S&P 500 미편입 인기 미국 주식 (데이트레이딩 활성 종목)
ACTIVE_US_STOCKS = [
    # 암호화폐/핀테크
    "MSTR", "COIN", "HOOD", "SOFI",
    # 전기차
    "RIVN", "LCID",
    # 암호화폐 채굴
    "MARA", "RIOT", "CLSK", "BITF",
    # AI/반도체 (S&P 500 미편입 또는 신규)
    "ARM", "SMCI",
    # 양자컴퓨팅
    "IONQ", "RGTI", "QUBT",
    # 밈주식
    "GME", "AMC",
    # 기타 고변동성
    "RKLB", "LUNR", "JOBY",
]

# Cache for S&P 500 data
_sp500_cache: Dict[str, List[str]] = {}


def fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """Wikipedia에서 S&P 500 종목 리스트 스크래핑"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
        df.columns = ["ticker", "name", "sector", "industry"]
        # 티커 정리 (BRK.B -> BRK-B 형식으로)
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df
    except Exception as e:
        print(f"Wikipedia 스크래핑 실패: {e}")
        return pd.DataFrame()


def get_sp500_by_sector() -> Dict[str, List[str]]:
    """섹터별 S&P 500 종목 반환"""
    global _sp500_cache

    if _sp500_cache:
        return _sp500_cache

    df = fetch_sp500_from_wikipedia()
    if df.empty:
        return {}

    result = {}
    for sector in df["sector"].unique():
        tickers = df[df["sector"] == sector]["ticker"].tolist()
        result[sector] = tickers

    _sp500_cache = result
    return result


def get_all_tickers() -> List[str]:
    """모든 S&P 500 티커 반환"""
    sectors = get_sp500_by_sector()
    all_tickers = []
    for tickers in sectors.values():
        all_tickers.extend(tickers)
    return sorted(list(set(all_tickers)))


def get_extended_tickers() -> List[str]:
    """S&P 500 + 레버리지 ETF + 인기 미국 주식 합산 (Ross Cameron 전략용)"""
    sp500 = get_all_tickers()
    extended = list(set(sp500 + LEVERAGED_ETFS + ACTIVE_US_STOCKS))
    return sorted(extended)


def get_ticker_info() -> pd.DataFrame:
    """티커별 상세 정보 (이름, 섹터, 산업) 반환"""
    return fetch_sp500_from_wikipedia()


# Alias for backward compatibility
SP500_SECTORS = SECTORS
