"""
PRISMA IA Trading Strategy
Estratégia de sinais para CALL/PUT em 1m OTC baseada em confluências e filtros
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import statistics
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient
from pocketoptionapi_async.models import TimeFrame


class SignalType(Enum):
    CALL = "CALL"
    PUT = "PUT"
    NONE = "NONE"


class ConfluenceType(Enum):
    LTB_BREAKOUT = "ltb_breakout"
    LTA_BREAKOUT = "lta_breakout"
    REJECTION = "rejection"
    SEQUENCE = "sequence"
    LATERAL_BREAK = "lateral_break"
    REVERSAL = "reversal"
    PRESSURE = "pressure"


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float

    @property
    def is_green(self) -> bool:
        return self.close > self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low

    @property
    def wick_upper(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def wick_lower(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def wick_ratio(self) -> float:
        if self.range_size == 0:
            return 0
        return max(self.wick_upper, self.wick_lower) / self.range_size


@dataclass
class SignalResult:
    signal: SignalType
    score: int
    confluences: List[ConfluenceType]
    asset: str
    timestamp: int
    price: float


class PrismaStrategy:
    def __init__(self, ssid: str, is_demo: bool = True):
        self.client = AsyncPocketOptionClient(ssid, is_demo=is_demo, persistent_connection=True)
        self.candles: List[Candle] = []
        self.asset: Optional[str] = None
        self.min_candles = 30  # mínimo para análise

    async def connect(self) -> bool:
        """Conecta à API"""
        return await self.client.connect()

    async def select_asset(self, asset: str):
        """Seleciona ativo OTC"""
        self.asset = asset
        logger.info(f"Ativo selecionado: {asset}")

    async def get_candles_history(self, count: int = 100):
        """Busca histórico de velas"""
        if not self.asset:
            return

        try:
            df = await self.client.get_candles_dataframe(self.asset, TimeFrame.MINUTE_1, count)
            if df is not None and not df.empty:
                self.candles = []
                for _, row in df.iterrows():
                    candle = Candle(
                        timestamp=int(row['timestamp']),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close'])
                    )
                    self.candles.append(candle)
                logger.info(f"Histórico carregado: {len(self.candles)} velas para {self.asset}")
        except Exception as e:
            logger.error(f"Erro ao buscar histórico: {e}")

    def calculate_trend_lines(self, candles: List[Candle], lookback: int = 20) -> tuple:
        """Calcula linhas de tendência simples (topos e fundos)"""
        if len(candles) < lookback:
            return None, None

        recent = candles[-lookback:]

        # Topos baixistas (resistência) - máximas descendentes
        highs = [c.high for c in recent]
        ltb = None
        if len(highs) >= 3:
            # Linha conectando os 2-3 últimos topos
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))

            if len(peaks) >= 2:
                # Linha entre os dois últimos picos
                p1, p2 = peaks[-2], peaks[-1]
                # Simplificado: média dos picos
                ltb = (p1[1] + p2[1]) / 2

        # Fundos altistas (suporte) - mínimas ascendentes
        lows = [c.low for c in recent]
        lta = None
        if len(lows) >= 3:
            valleys = []
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    valleys.append((i, lows[i]))

            if len(valleys) >= 2:
                v1, v2 = valleys[-2], valleys[-1]
                lta = (v1[1] + v2[1]) / 2

        return ltb, lta

    def check_ltb_breakout(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Verifica rompimento de LTB (linha de topos baixistas)"""
        if len(candles) < 5:
            return False, SignalType.NONE

        ltb, _ = self.calculate_trend_lines(candles, 20)
        if ltb is None:
            return False, SignalType.NONE

        current = candles[-1]
        previous = candles[-2]

        # Rompimento: vela fecha acima da LTB
        if current.close > ltb and previous.close <= ltb:
            return True, SignalType.CALL

        return False, SignalType.NONE

    def check_lta_breakout(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Verifica rompimento de LTA (linha de fundos altistas)"""
        if len(candles) < 5:
            return False, SignalType.NONE

        _, lta = self.calculate_trend_lines(candles, 20)
        if lta is None:
            return False, SignalType.NONE

        current = candles[-1]
        previous = candles[-2]

        # Rompimento: vela fecha abaixo da LTA
        if current.close < lta and previous.close >= lta:
            return True, SignalType.PUT

        return False, SignalType.NONE

    def detect_rejection(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Detecta rejeição com wick grande"""
        if len(candles) < 3:
            return False, SignalType.NONE

        current = candles[-1]

        # Wick > 60% do range
        if current.wick_ratio > 0.6:
            if current.wick_lower > current.wick_upper and current.is_green:
                return True, SignalType.CALL  # Rejeição em suporte
            elif current.wick_upper > current.wick_lower and not current.is_green:
                return True, SignalType.PUT   # Rejeição em resistência

        return False, SignalType.NONE

    def detect_sequence(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Detecta sequência forte de 4+ velas na mesma direção"""
        if len(candles) < 10:
            return False, SignalType.NONE

        recent = candles[-10:]
        green_count = sum(1 for c in recent if c.is_green)
        red_count = len(recent) - green_count

        if green_count >= 4 and green_count > red_count:
            return True, SignalType.CALL
        elif red_count >= 4 and red_count > green_count:
            return True, SignalType.PUT

        return False, SignalType.NONE

    def calculate_pressure_score(self, candles: List[Candle]) -> float:
        """Calcula score de pressão (compra vs venda)"""
        if len(candles) < 10:
            return 0

        recent = candles[-10:]
        pressure = 0

        for c in recent:
            if c.is_green:
                # Pressão de compra: body grande, wick pequeno
                body_ratio = c.body_size / c.range_size if c.range_size > 0 else 0
                pressure += body_ratio * 10
            else:
                # Pressão de venda
                body_ratio = c.body_size / c.range_size if c.range_size > 0 else 0
                pressure -= body_ratio * 10

        return pressure / len(recent)

    def detect_pressure_imbalance(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Detecta pressão desequilibrada"""
        pressure = self.calculate_pressure_score(candles)

        if pressure > 7:  # Muito alta pressão de compra
            return True, SignalType.CALL
        elif pressure < -7:  # Muito alta pressão de venda
            return True, SignalType.PUT

        return False, SignalType.NONE

    def is_lateralization(self, candles: List[Candle]) -> bool:
        """Detecta lateralização/consolidação"""
        if len(candles) < 20:
            return False

        recent = candles[-20:]
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]

        range_total = max(highs) - min(lows)
        avg_price = sum(c.close for c in recent) / len(recent)

        # Range < 0.05% do preço
        return range_total / avg_price < 0.0005

    def detect_lateral_break(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Detecta quebra de lateralização"""
        if not self.is_lateralization(candles[:-1]):  # Estava lateral antes
            return False, SignalType.NONE

        current = candles[-1]
        avg_price = sum(c.close for c in candles[-21:-1]) / 20

        # Vela grande rompe lateral
        if abs(current.close - avg_price) / avg_price > 0.0005:
            return True, SignalType.CALL if current.close > avg_price else SignalType.PUT

        return False, SignalType.NONE

    def detect_reversal(self, candles: List[Candle]) -> tuple[bool, SignalType]:
        """Detecta vela de reversão isolada"""
        if len(candles) < 5:
            return False, SignalType.NONE

        current = candles[-1]
        prev_3 = candles[-4:-1]

        # Wick > 50%
        if current.wick_ratio < 0.5:
            return False, SignalType.NONE

        # Apenas 1 vela contra nas últimas 3
        opposite_count = sum(1 for c in prev_3 if c.is_green != current.is_green)
        if opposite_count > 1:
            return False, SignalType.NONE

        # Confirmação na próxima vela (simulado com vela atual)
        if current.wick_lower > current.wick_upper and current.is_green:
            return True, SignalType.CALL
        elif current.wick_upper > current.wick_lower and not current.is_green:
            return True, SignalType.PUT

        return False, SignalType.NONE

    def detect_trend_context(self, candles: List[Candle]) -> Optional[SignalType]:
        """Detecta contexto de tendência (filtro)"""
        if len(candles) < 30:
            return None

        recent = candles[-30:]
        up_moves = sum(1 for c in recent if c.close > c.open)
        down_moves = len(recent) - up_moves

        if up_moves > down_moves * 1.8:
            return SignalType.CALL  # Tendência de alta
        elif down_moves > up_moves * 1.8:
            return SignalType.PUT   # Tendência de baixa

        return None  # Neutro

    def analyze_signal(self) -> SignalResult:
        """Analisa todas as confluências e gera sinal"""
        if len(self.candles) < self.min_candles or not self.asset:
            return SignalResult(SignalType.NONE, 0, [], self.asset or "", 0, 0)

        score = 0
        confluences = []
        signal_type = SignalType.NONE

        # Verifica cada confluência
        checks = [
            (self.check_ltb_breakout, 45, ConfluenceType.LTB_BREAKOUT),
            (self.check_lta_breakout, 45, ConfluenceType.LTA_BREAKOUT),
            (self.detect_rejection, 40, ConfluenceType.REJECTION),
            (self.detect_sequence, 35, ConfluenceType.SEQUENCE),
            (self.detect_lateral_break, 35, ConfluenceType.LATERAL_BREAK),
            (self.detect_reversal, 30, ConfluenceType.REVERSAL),
            (self.detect_pressure_imbalance, 25, ConfluenceType.PRESSURE),
        ]

        for check_func, weight, conf_type in checks:
            detected, sig = check_func(self.candles)
            if detected:
                score += weight
                confluences.append(conf_type)
                if signal_type == SignalType.NONE:
                    signal_type = sig
                elif signal_type != sig:
                    # Conflito de sinais, aborta
                    return SignalResult(SignalType.NONE, 0, [], self.asset, self.candles[-1].timestamp, self.candles[-1].close)

        # Verifica pressão adicional se sequência detectada
        if ConfluenceType.SEQUENCE in confluences:
            pressure = self.calculate_pressure_score(self.candles)
            if (signal_type == SignalType.CALL and pressure > 7) or (signal_type == SignalType.PUT and pressure < -7):
                score += 10  # Bônus por pressão alta

        # Filtro de tendência
        trend = self.detect_trend_context(self.candles)
        if trend and signal_type != trend:
            # Contra tendência, aborta
            return SignalResult(SignalType.NONE, 0, [], self.asset, self.candles[-1].timestamp, self.candles[-1].close)

        # Verifica condições mínimas
        if score >= 90 and len(set(confluences)) >= 3:
            return SignalResult(signal_type, score, confluences, self.asset, self.candles[-1].timestamp, self.candles[-1].close)

    async def place_order(self, signal_result: SignalResult, amount: float = 1.0):
        """Coloca ordem baseada no sinal"""
        try:
            # Converte sinal para direção
            direction = "call" if signal_result.signal == SignalType.CALL else "put"

            # Coloca ordem
            order_result = await self.client.place_order(
                active=signal_result.asset,
                direction=direction,
                amount=amount,
                timeframe=1  # 1 minuto
            )

            if order_result:
                logger.success(f"Ordem colocada: {direction.upper()} {signal_result.asset} | Amount: {amount}")
            else:
                logger.error("Falha ao colocar ordem")

        except Exception as e:
            logger.error(f"Erro ao colocar ordem: {e}")
        """Executa a estratégia em loop com WebSocket"""
        logger.info("Iniciando estratégia PRISMA IA")

        if not await self.connect():
            logger.error("Falha na conexão")
            return

        # Carrega histórico inicial
        await self.get_candles_history(100)

        # Configura callback para novas velas
        async def on_candle_update(candle_data):
            """Callback chamado quando nova vela chega"""
            if not self.asset or candle_data.get('asset') != self.asset:
                return

            # Adiciona nova vela
            new_candle = Candle(
                timestamp=candle_data['timestamp'],
                open=candle_data['open'],
                high=candle_data['high'],
                low=candle_data['low'],
                close=candle_data['close']
            )

            self.candles.append(new_candle)
            if len(self.candles) > 200:  # Mantém máximo
                self.candles.pop(0)

            # Analisa sinal
            result = self.analyze_signal()

            if result.signal != SignalType.NONE:
                logger.success(f"SINAL {result.signal.value} | Score: {result.score} | Confluências: {[c.value for c in result.confluences]} | Ativo: {result.asset} | Preço: {result.price}")

                # Coloca ordem
                await self.place_order(result)

        # Adiciona callback (assumindo que o client suporta)
        # self.client.add_candle_callback(on_candle_update)

        # Por enquanto, simula loop
        logger.info("Estratégia rodando... (simulado)")
        while True:
            await asyncio.sleep(60)
            # Em produção, o callback seria chamado pelo WebSocket


async def main():
    # Exemplo de uso
    SSID = r'42["auth",{"session":"demo_session","isDemo":1,"uid":12345,"platform":1}]'  # Substitua pelo seu SSID

    strategy = PrismaStrategy(SSID, is_demo=True)

    # Seleciona ativo
    await strategy.select_asset("EURUSD_otc")

    # Executa estratégia
    await strategy.run_strategy()


if __name__ == "__main__":
    asyncio.run(main())