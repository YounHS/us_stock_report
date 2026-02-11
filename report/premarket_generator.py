"""프리마켓 리포트 생성기"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
import logging

from jinja2 import Environment, FileSystemLoader

from config.settings import settings

logger = logging.getLogger(__name__)


class PreMarketReportGenerator:
    """프리마켓 HTML 리포트 생성기"""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True,
        )

    def generate(
        self,
        market_premarket: Dict,
        sector_premarket: Dict,
        significant_movers: Dict,
        previous_recommendations: List[Dict],
        economic_calendar: Optional[Dict] = None,
        earnings_calendar: Optional[Dict] = None,
        news: Optional[Dict] = None,
        report_date: Optional[str] = None,
    ) -> str:
        """
        프리마켓 리포트 HTML 생성

        Args:
            market_premarket: 시장 ETF 프리마켓 데이터 {ticker: PreMarketData as dict}
            sector_premarket: 섹터 ETF 프리마켓 데이터 {ticker: PreMarketData as dict}
            significant_movers: {"gainers": [...], "losers": [...]}
            previous_recommendations: 전일 추천 종목 + PM 데이터 + 기술적 분석 리스트
            economic_calendar: 경제 캘린더
            earnings_calendar: 실적 캘린더
            news: 뉴스 데이터
            report_date: 리포트 날짜

        Returns:
            HTML 문자열
        """
        template = self.env.get_template("premarket_report.html")

        tz = ZoneInfo(settings.general.timezone)
        now = datetime.now(tz)
        context = {
            "report_date": report_date or now.strftime("%Y-%m-%d"),
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "market_premarket": market_premarket,
            "sector_premarket": sector_premarket,
            "significant_movers": significant_movers,
            "previous_recommendations": previous_recommendations,
            "economic_calendar": economic_calendar,
            "earnings_calendar": earnings_calendar,
            "news": news,
        }

        return template.render(**context)

    def save_to_file(self, html: str, filepath: Optional[str] = None) -> str:
        """
        리포트를 파일로 저장

        Args:
            html: HTML 문자열
            filepath: 저장 경로 (없으면 자동 생성)

        Returns:
            저장된 파일 경로
        """
        if filepath is None:
            tz = ZoneInfo(settings.general.timezone)
            date_str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
            filepath = f"premarket_report_{date_str}.html"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"프리마켓 리포트 저장 완료: {filepath}")
        return filepath
