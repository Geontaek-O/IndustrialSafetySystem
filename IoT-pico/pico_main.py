import bluetooth
import struct
import time
import math
import machine
import dht
import uasyncio as asyncio
import _thread
from micropython import const

# ================================================================
# 하드웨어 핀 매핑 및 센서 임계값 설정
# ================================================================
DHT_PIN = machine.Pin(15, machine.Pin.IN, machine.Pin.PULL_UP)
I2S_SCK = machine.Pin(2)
I2S_WS  = machine.Pin(3)
I2S_SD  = machine.Pin(4)
led     = machine.Pin("LED", machine.Pin.OUT)

VIB_ADC = machine.ADC(26)
VIB_THRESHOLD = 10000

# 진동 센서 디바운싱(오작동 방지) 설정
VIB_DEBOUNCE_WINDOW_MS = 1000   # 유효 판정 시간 창 (1초)
VIB_TRIGGER_COUNT      = 3      # 해당 시간 내 누적 발생 요구 횟수

# ================================================================
# 멀티코어 공유 상태 변수
# ================================================================
shared_max_amp = 0
shared_vib_val = 0
is_mic_running = False
mic_pcm_chunk  = None    # 안드로이드 전송용 PCM 버퍼
mic_chunk_lock = _thread.allocate_lock()

mic_thread_alive = False # Core 1 스레드 실제 동작 상태

# 상태 표시 LED 제어 플래그
led_sos_mode = False   # 충격 감지 시 특수 점멸 패턴
led_stt_mode = False   # STT 오디오 전송 중 점등 유지

# ================================================================
# BLE UUID 규격 (안드로이드 MonitoringService와 1:1 매칭 필수)
# ================================================================
_SERVICE_UUID = bluetooth.UUID("12345678-1234-1234-1234-123456789abc")
_DHT22_UUID   = bluetooth.UUID("12345678-1234-1234-1234-123456789001")
_MIC_UUID     = bluetooth.UUID("12345678-1234-1234-1234-123456789002")
_VIB_UUID     = bluetooth.UUID("12345678-1234-1234-1234-123456789003")

# BLE GATT 특성(Characteristic) 속성 플래그
_FLAG_READ_NOTIFY = const(0x12)
_FLAG_NOTIFY      = const(0x10)

# ================================================================
# 마이크 전송 설정
# ================================================================
MIC_CHUNK_SIZE = 20                 # BLE MTU 최적화 전송 단위
MIC_AMPLITUDE_THRESHOLD = 500       # 무음 필터링 컷오프 (이하 값 폐기)

# ================================================================
# BLE 제어 클래스
# ================================================================
class SafetyBLE:
    def __init__(self):
        self._ble = bluetooth.BLE()
        self._ble.active(True)
        self._ble.irq(self._irq)

        # GATT 서버 서비스 및 3개 특성(DHT22, MIC, VIB) 등록
        h = self._ble.gatts_register_services((
            (_SERVICE_UUID, (
                (_DHT22_UUID, _FLAG_READ_NOTIFY),  # 온습도
                (_MIC_UUID,   _FLAG_NOTIFY),       # 마이크 PCM
                (_VIB_UUID,   _FLAG_NOTIFY),       # 충격 알림
            )),
        ))[0]

        self._dht_h = h[0]
        self._mic_h = h[1]
        self._vib_h = h[2]

        self.is_connected = False
        self.conn_handle  = None
        self._advertise()
        print("BLE 광고 시작: IoTNode_A3F")

    # BLE 상태 변경 인터럽트 콜백
    def _irq(self, event, data):
        if event == 1:   # 연결 성공
            self.conn_handle  = data[0]
            self.is_connected = True
            print("\n[BLE] 앱 연결 완료. handle={}".format(self.conn_handle))
        elif event == 2: # 연결 해제
            self.conn_handle  = None
            self.is_connected = False
            print("\n[BLE] 연결 해제. 대기 모드 전환.")
            self._advertise()

    # BLE Advertising (기기 검색 허용) 시작
    def _advertise(self):
        name = b"IoTNode_A3F"
        adv  = b"\x02\x01\x06" + bytes([len(name) + 1, 0x09]) + name
        self._ble.gap_advertise(500_000, adv, connectable=True)

    # 온습도 Notify
    def notify_dht22(self, temp, humid):
        if not self.is_connected:
            return
        try:
            msg = "{:.1f},{:.1f}".format(temp, humid).encode()
            self._ble.gatts_notify(self.conn_handle, self._dht_h, msg)
        except Exception as e:
            print("\n[BLE] DHT22 전송 오류: {}".format(e))

    # 오디오 청크 Notify (MTU 한계로 인한 분할 전송)
    def notify_mic(self, pcm_bytes):
        if not self.is_connected:
            return
        for i in range(0, len(pcm_bytes), MIC_CHUNK_SIZE):
            chunk = pcm_bytes[i:i + MIC_CHUNK_SIZE]
            try:
                self._ble.gatts_notify(self.conn_handle, self._mic_h, chunk)
                time.sleep_ms(10)  # 패킷 누락 방지용 송신 딜레이
            except Exception as e:
                print("\n[BLE] MIC 전송 오류: {}".format(e))
                break

    # 충격 감지 Notify
    def notify_vib(self):
        if not self.is_connected:
            return
        try:
            self._ble.gatts_notify(self.conn_handle, self._vib_h, b"1")
        except Exception as e:
            print("\n[BLE] VIB 전송 오류: {}".format(e))


# ================================================================
# [Core 1] I2S 마이크 데이터 수집 스레드 (실시간성 보장)
# ================================================================
def mic_core1_worker():
    global shared_max_amp, is_mic_running, mic_pcm_chunk, mic_thread_alive
    mic_thread_alive = True
    audio_in = None
    try:
        # I2S 하드웨어 초기화 (16kHz, 16bit, Mono)
        audio_in = machine.I2S(
            0,
            sck    = I2S_SCK,
            ws     = I2S_WS,
            sd     = I2S_SD,
            mode   = machine.I2S.RX,
            bits   = 16,
            format = machine.I2S.MONO,
            rate   = 16000,
            ibuf   = 2048
        )
        print("\n[Core 1] 마이크 하드웨어 활성화")

        read_buf = bytearray(1024)   # 읽기 버퍼 (1024 bytes)
        while is_mic_running:
            n = audio_in.readinto(read_buf) # Blocking 읽기
            if n > 0:
                # 진폭(소리 크기) 모니터링
                samples = struct.unpack('<' + 'h' * (n // 2), read_buf[:n])
                shared_max_amp = max(abs(s) for s in samples)

                # 공유 청크 업데이트 (Lock 적용)
                with mic_chunk_lock:
                    mic_pcm_chunk = bytes(read_buf[:n])

    except Exception as e:
        print("\n[Core 1] 오류: {}".format(e))
    finally:
        # 자원 해제 및 종료 상태 동기화
        if audio_in:
            audio_in.deinit()
            print("\n[Core 1] 마이크 자원 해제 완료")
        mic_thread_alive = False


# ================================================================
# [Core 0] 메인 컨트롤러 (센서 관리 및 스레드 생명주기)
# ================================================================
async def main_controller(ble):
    global is_mic_running, shared_max_amp, mic_pcm_chunk, shared_vib_val
    global mic_thread_alive
    sensor = None

    while True:
        # 1. BLE 미연결 시 자원 정리 및 대기
        if not ble.is_connected:
            if is_mic_running:
                print("\n[시스템] 연결 해제 — 리소스 정리")
                is_mic_running = False
                sensor         = None
                shared_max_amp = 0
                mic_pcm_chunk  = None
            await asyncio.sleep(1)
            continue

        # 2. BLE 최초 연결 시 하드웨어 활성화
        if sensor is None:
            print("\n[시스템] 센서 및 마이크 초기화 시작...")
            
            # 이전 마이크 스레드 완전 종료 대기 (Race Condition 방어)
            wait_max = 50
            while mic_thread_alive and wait_max > 0:
                await asyncio.sleep_ms(100)
                wait_max -= 1
            if mic_thread_alive:
                print("\n[시스템] ⚠️ 이전 마이크 스레드 종료 대기 시간 초과 — 재시도")
                await asyncio.sleep(1)
                continue

            # 센서 초기화 및 Core 1 스레드 시작
            sensor = dht.DHT22(DHT_PIN)
            if not is_mic_running and not mic_thread_alive:
                is_mic_running = True
                _thread.start_new_thread(mic_core1_worker, ())
            print("[시스템] 모든 장치 활성화 완료")

        # 3. DHT22 데이터 획득 및 전송
        try:
            sensor.measure()
            t, h = sensor.temperature(), sensor.humidity()
            ble.notify_dht22(t, h)

            # 콘솔 모니터링 로그 (제자리 덮어쓰기)
            print("\r🌡️ {:.1f}C, {:.1f}% | 🎤 Mic: {:>5} | 🚨 Vib: {:>5}".format(
                t, h, shared_max_amp, shared_vib_val), end="")

        except Exception as e:
            print("\n[DHT22] 측정 오류: {}".format(e))
            await asyncio.sleep(1)
            continue

        await asyncio.sleep(3)


# 오디오 PCM RMS(Root Mean Square) 계산 유틸리티
def calc_rms(buf, n):
    if n < 2:
        return 0
    samples = struct.unpack('<' + 'h' * (n // 2), buf[:n])
    mean_sq = sum(s * s for s in samples) / len(samples)
    return math.sqrt(mean_sq)


# ================================================================
# 마이크 오디오 전송 루프 (무음 필터링 기반 통신 최적화)
# ================================================================
async def mic_sender(ble):
    global mic_pcm_chunk, led_stt_mode
    while True:
        if ble.is_connected:
            with mic_chunk_lock:
                chunk = mic_pcm_chunk
                mic_pcm_chunk = None
                
            if chunk:
                # 소리 크기 측정 후 임계값 초과 시에만 전송
                rms = calc_rms(chunk, len(chunk))
                if rms > MIC_AMPLITUDE_THRESHOLD:
                    led_stt_mode = True    # STT 전송 표시 LED 켬
                    ble.notify_mic(chunk)
                    led_stt_mode = False   # 전송 완료 후 LED 원복
        await asyncio.sleep_ms(50)


# ================================================================
# 상태 표시 LED 제어 (디버그 및 현장 확인용)
# - BLE 연결됨   : 0.2초 간격 빠른 깜빡임
# - BLE 미연결   : 1.5초 간격 느린 깜빡임 (광고 중)
# - 충격 감지    : 빠른 3회 점멸 후 복귀
# - STT 전송 중  : 점등 유지 (우선순위 최상)
# ================================================================
async def heartbeat(ble):
    global led_sos_mode, led_stt_mode
    while True:
        if led_stt_mode:
            led.on()
            await asyncio.sleep_ms(100)

        elif led_sos_mode:
            for _ in range(3):
                led.on();  await asyncio.sleep_ms(80)
                led.off(); await asyncio.sleep_ms(80)
            led_sos_mode = False
            await asyncio.sleep_ms(300)

        elif ble.is_connected:
            led.toggle()
            await asyncio.sleep_ms(200)

        else:
            led.toggle()
            await asyncio.sleep_ms(1500)


# ================================================================
# 진동 센서 감지 루프 (소프트웨어 디바운싱 적용)
# ================================================================
async def vib_sender(ble):
    global shared_vib_val, led_sos_mode

    # 최근 트리거 기록 큐 (타임스탬프)
    recent_triggers = [] 

    while True:
        current_val = VIB_ADC.read_u16()
        shared_vib_val = current_val 

        if current_val > VIB_THRESHOLD:
            now_ms = time.ticks_ms()
            
            # 유효 시간(1초) 지난 트리거 기록 소거
            recent_triggers = [
                t for t in recent_triggers
                if time.ticks_diff(now_ms, t) < VIB_DEBOUNCE_WINDOW_MS
            ]
            recent_triggers.append(now_ms)

            # 1초 내 목표 횟수(3회) 누적 시 SOS 상황 확정
            if len(recent_triggers) >= VIB_TRIGGER_COUNT:
                led_sos_mode = True   
                if ble.is_connected:
                    ble.notify_vib()
                    print("\n[VIB] ⚠️ 충격 감지! ({}회/1초) → SOS 전송".format(
                        len(recent_triggers)))
                    recent_triggers.clear()
                    await asyncio.sleep(2)   # 알림 폭주 방지용 쿨다운
                else:
                    print("\n[VIB] ⚠️ 충격 감지! BLE 미연결로 전송 보류")
                    recent_triggers.clear()
                    await asyncio.sleep_ms(500) 
                continue

        await asyncio.sleep_ms(20)


# ================================================================
# 메인 실행 부 (비동기 이벤트 루프 구동)
# ================================================================
async def main():
    ble = SafetyBLE()
    print("=== 시스템 가동 ===")
    
    # 4개의 비동기 코루틴 병렬 실행
    await asyncio.gather(
        main_controller(ble),
        mic_sender(ble),
        heartbeat(ble),
        vib_sender(ble)
    )

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n[시스템] 사용자에 의해 중단")
except Exception as e:
    print("\n치명적 오류: {}".format(e))