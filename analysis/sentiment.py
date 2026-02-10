"""뉴스 감성 분석 모듈 (VADER 기반)"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsItemSentiment:
    """개별 뉴스 감성 분석 결과"""
    title: str
    link: str
    compound: float  # -1.0 ~ 1.0
    label: str  # positive / negative / neutral


@dataclass
class TickerSentiment:
    """종목별 감성 집계 결과"""
    ticker: str
    avg_compound: float  # 평균 compound score
    positive_count: int
    negative_count: int
    neutral_count: int
    total_count: int
    gauge_score: float  # 0-100 (50=중립)
    label: str  # 매우 긍정 / 긍정 / 중립 / 부정 / 매우 부정
    news_items: List[NewsItemSentiment] = field(default_factory=list)


class NewsSentimentAnalyzer:
    """VADER 기반 뉴스 헤드라인 감성 분석기"""

    def __init__(self):
        self._analyzer = None

    def _get_analyzer(self):
        """VADER SentimentIntensityAnalyzer lazy 초기화"""
        if self._analyzer is None:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self._analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                logger.info("VADER lexicon 다운로드 중...")
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def _classify(self, compound: float) -> str:
        """compound score를 라벨로 분류"""
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"

    def _to_korean_label(self, avg_compound: float) -> str:
        """평균 compound를 5단계 한국어 라벨로 변환"""
        if avg_compound >= 0.35:
            return "매우 긍정"
        elif avg_compound >= 0.05:
            return "긍정"
        elif avg_compound > -0.05:
            return "중립"
        elif avg_compound > -0.35:
            return "부정"
        return "매우 부정"

    def _compound_to_gauge(self, avg_compound: float) -> float:
        """compound (-1~1)을 게이지 점수 (0~100)로 변환"""
        return round((avg_compound + 1) / 2 * 100, 1)

    def analyze_ticker_news(
        self, ticker: str, news_items: List[Dict]
    ) -> Optional[TickerSentiment]:
        """종목 뉴스 감성 분석

        Args:
            ticker: 종목 티커
            news_items: [{"title": ..., "link": ..., ...}, ...]

        Returns:
            TickerSentiment 또는 뉴스 없으면 None
        """
        if not news_items:
            return None

        analyzer = self._get_analyzer()
        sentiments: List[NewsItemSentiment] = []
        pos = neg = neu = 0

        for item in news_items:
            title = item.get("title", "")
            if not title:
                continue
            scores = analyzer.polarity_scores(title)
            compound = scores["compound"]
            label = self._classify(compound)

            sentiments.append(NewsItemSentiment(
                title=title,
                link=item.get("link", ""),
                compound=compound,
                label=label,
            ))

            if label == "positive":
                pos += 1
            elif label == "negative":
                neg += 1
            else:
                neu += 1

        if not sentiments:
            return None

        avg_compound = sum(s.compound for s in sentiments) / len(sentiments)

        return TickerSentiment(
            ticker=ticker,
            avg_compound=round(avg_compound, 4),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            total_count=len(sentiments),
            gauge_score=self._compound_to_gauge(avg_compound),
            label=self._to_korean_label(avg_compound),
            news_items=sentiments,
        )

    def analyze_multiple_tickers(
        self, ticker_news_map: Dict[str, List[Dict]]
    ) -> Dict[str, TickerSentiment]:
        """여러 종목 일괄 감성 분석

        Args:
            ticker_news_map: {ticker: [news_items]}

        Returns:
            {ticker: TickerSentiment}
        """
        results = {}
        for ticker, news_items in ticker_news_map.items():
            result = self.analyze_ticker_news(ticker, news_items)
            if result:
                results[ticker] = result
        return results
