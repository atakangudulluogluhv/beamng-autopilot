"""
BeamNG -> M5StickC Plus Telemetry Bridge
-----------------------------------------
Reads OutGauge UDP packets from BeamNG.drive and forwards
RPM, speed, and gear to the M5StickC Plus over WiFi UDP.

BeamNG OutGauge setup (do this once):
  1. Open BeamNG.drive
  2. Go to Options -> Other -> OutGauge
  3. Enable OutGauge, set IP to 127.0.0.1, port 4444, interval ~20ms
  OR manually create/edit:
     %APPDATA%/BeamNG.drive/0.xx/settings/outgauge.json
     {
       "host": "127.0.0.1",
       "port": 4444,
       "delay": 0.02
     }

Usage:
  1. Find your M5StickC Plus IP (shown on its screen at boot)
  2. Set M5STICK_IP below
  3. Start BeamNG, load a vehicle
  4. Run: python beamng_telemetry.py
"""

import socket
import struct
import time
import sys

# Configuration

BEAMNG_LISTEN_HOST = "0.0.0.0"   # Listen on all interfaces
BEAMNG_LISTEN_PORT = 4444         # Must match BeamNG OutGauge port setting

M5STICK_IP   = "192.168.1.XXX"   # <-- Change to your M5StickC Plus IP address
M5STICK_PORT = 5555

SEND_RATE_HZ  = 30                # Max updates per second to M5Stick
RECV_TIMEOUT  = 0.5               # Seconds to wait for BeamNG packet

# OutGauge Packet Layout
#  Matches LFS / BeamNG OutGauge spec (little-endian)
#
#  Offset  Size  Field
#  0       4     Time        (ms, uint32)
#  4       4     Car[4]      (char[4])
#  8       2     Flags       (uint16)
#  10      1     Gear        (uint8)  0=Rev, 1=N, 2=1st, 3=2nd ...
#  11      1     PLID        (uint8)
#  12      4     Speed       (float, m/s)
#  16      4     RPM         (float)
#  20      4     Turbo       (float, bar)
#  24      4     EngTemp     (float, °C)
#  28      4     Fuel        (float, 0-1)
#  32      4     OilPressure (float, bar)
#  36      4     OilTemp     (float, °C)
#  40      4     DashLights  (uint32)
#  44      4     ShowLights  (uint32)
#  48      4     Throttle    (float, 0-1)
#  52      4     Brake       (float, 0-1)
#  56      4     Clutch      (float, 0-1)
#  60      16    Display1    (char[16])
#  76      16    Display2    (char[16])
#  92      4     ID          (int32, optional)

OUTGAUGE_FMT = "<I4sHBBfffffffIIfff16s16s"
OUTGAUGE_MIN_SIZE = struct.calcsize(OUTGAUGE_FMT)  # 92 bytes

GEAR_NAMES = {0: "R", 1: "N", 2: "1", 3: "2", 4: "3",
              5: "4", 6: "5", 7: "6", 8: "7", 9: "8"}


def parse_outgauge(data: bytes) -> dict | None:
    """Parse a raw OutGauge UDP packet. Returns dict or None on error."""
    if len(data) < OUTGAUGE_MIN_SIZE:
        return None
    try:
        fields = struct.unpack_from(OUTGAUGE_FMT, data)
        _, car, flags, gear, plid, speed, rpm, turbo, eng_temp, fuel, \
            oil_press, oil_temp, dash_lights, show_lights, \
            throttle, brake, clutch, disp1, disp2 = fields

        return {
            "speed_ms":  speed,
            "speed_kmh": speed * 3.6,
            "rpm":       max(0.0, rpm),
            "gear":      gear,
            "gear_str":  GEAR_NAMES.get(gear, str(gear - 1)),
            "throttle":  throttle,
            "brake":     brake,
            "fuel":      fuel,
            "eng_temp":  eng_temp,
        }
    except struct.error:
        return None


def format_packet(data: dict) -> bytes:
    """
    Compact CSV packet sent to M5StickC Plus:
      RPM,SPEED_KMH,GEAR_STR,THROTTLE,BRAKE\n
    All values are integers / short strings to save bandwidth.
    """
    msg = (
        f"{int(data['rpm'])},"
        f"{int(data['speed_kmh'])},"
        f"{data['gear_str']},"
        f"{int(data['throttle'] * 100)},"
        f"{int(data['brake'] * 100)}\n"
    )
    return msg.encode()


def main():
    if M5STICK_IP == "192.168.1.XXX":
        print("ERROR: Set M5STICK_IP in the script before running.")
        sys.exit(1)

    # Receive socket (from BeamNG)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.bind((BEAMNG_LISTEN_HOST, BEAMNG_LISTEN_PORT))
    recv_sock.settimeout(RECV_TIMEOUT)

    # Send socket (to M5StickC Plus)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    min_interval = 1.0 / SEND_RATE_HZ
    last_send    = 0.0
    packets_recv = 0
    packets_sent = 0

    print(f"[BeamNG Telemetry Bridge]")
    print(f"  Listening : {BEAMNG_LISTEN_HOST}:{BEAMNG_LISTEN_PORT}")
    print(f"  Forwarding: {M5STICK_IP}:{M5STICK_PORT}")
    print(f"  Send rate : {SEND_RATE_HZ} Hz\n")
    print("Waiting for BeamNG OutGauge data...  (Ctrl+C to quit)\n")

    try:
        while True:
            try:
                raw, _ = recv_sock.recvfrom(256)
            except socket.timeout:
                continue

            packets_recv += 1
            telemetry = parse_outgauge(raw)
            if telemetry is None:
                continue

            now = time.monotonic()
            if now - last_send >= min_interval:
                pkt = format_packet(telemetry)
                send_sock.sendto(pkt, (M5STICK_IP, M5STICK_PORT))
                last_send = now
                packets_sent += 1

            # Status line every 100 packets
            if packets_recv % 100 == 0:
                print(
                    f"\r  RPM: {int(telemetry['rpm']):5d}  "
                    f"Speed: {int(telemetry['speed_kmh']):3d} km/h  "
                    f"Gear: {telemetry['gear_str']}  "
                    f"(recv={packets_recv} sent={packets_sent})",
                    end="", flush=True
                )

    except KeyboardInterrupt:
        print(f"\n\nStopped. Received {packets_recv} packets, sent {packets_sent}.")
    finally:
        recv_sock.close()
        send_sock.close()


if __name__ == "__main__":
    main()
