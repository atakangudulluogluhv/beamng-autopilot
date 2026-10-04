/*
  BeamNG Dashboard for M5StickC Plus
  =====================================
  Displays RPM gauge, speed, and gear in a dark racing-style UI.

  Hardware : M5StickC Plus (ESP32-PICO, 135x240 TFT, landscape → 240x135)
  Libraries: M5StickCPlus  (install via Arduino Library Manager)

  Setup:
    1. Install "M5StickCPlus" library in Arduino IDE
    2. Set WIFI_SSID / WIFI_PASS below
    3. Flash to M5StickC Plus
    4. The IP address is shown on screen — enter it in beamng_telemetry.py
    5. Run the Python script, then launch BeamNG with OutGauge enabled
*/

#include <M5StickCPlus.h>
#include <WiFi.h>
#include <WiFiUDP.h>

// Config
const char* WIFI_SSID = "your-wifi-ssid";
const char* WIFI_PASS = "your-wifi-password";
const int   UDP_PORT  = 5555;
const int   MAX_RPM   = 10000;               // Adjust to your car's redline

// Screen dimensions (landscape)
#define SCR_W 240
#define SCR_H 135

// Layout constants
#define TITLE_H   18
#define SPEED_Y   20
#define GEAR_X    190
#define GEAR_Y    20
#define RPM_BAR_X 10
#define RPM_BAR_Y 90
#define RPM_BAR_W 220
#define RPM_BAR_H 22

// Colour palette
#define C_BG        TFT_BLACK
#define C_ACCENT    0x04FF    // Cyan-ish
#define C_DIM       0x4208    // Dark gray
#define C_SPEED     TFT_WHITE
#define C_UNIT      0xC618    // Light gray
#define C_GEAR      0xFFE0    // Yellow
#define C_RPM_LO    0x07E0    // Green
#define C_RPM_MID   0xFFE0    // Yellow
#define C_RPM_HI    0xF800    // Red
#define C_RPM_PEAK  0xF81F    // Magenta (limiter zone)
#define C_DIVIDER   0x2104    // Very dark gray

// State
int  g_rpm      = 0;
int  g_speed    = 0;
char g_gear[3]  = "N";
int  g_throttle = 0;
int  g_brake    = 0;

int  g_prev_rpm      = -1;
int  g_prev_speed    = -1;
char g_prev_gear[3]  = "";
bool g_prev_brake    = false;

WiFiUDP udp;
char    packetBuf[128];

// Helpers

// Linear colour interpolation between two 16-bit 565 colours
uint16_t lerpColor(uint16_t c1, uint16_t c2, float t) {
  uint8_t r1 = (c1 >> 11) & 0x1F, g1 = (c1 >> 5) & 0x3F, b1 = c1 & 0x1F;
  uint8_t r2 = (c2 >> 11) & 0x1F, g2 = (c2 >> 5) & 0x3F, b2 = c2 & 0x1F;
  uint8_t r  = r1 + (int)((r2 - r1) * t);
  uint8_t g  = g1 + (int)((g2 - g1) * t);
  uint8_t b  = b1 + (int)((b2 - b1) * t);
  return (r << 11) | (g << 5) | b;
}

// Colour for a given RPM ratio (0.0 → 1.0)
uint16_t rpmColor(float ratio) {
  if (ratio < 0.60f) return lerpColor(C_RPM_LO,  C_RPM_MID, ratio / 0.60f);
  if (ratio < 0.82f) return lerpColor(C_RPM_MID, C_RPM_HI,  (ratio - 0.60f) / 0.22f);
  return C_RPM_PEAK;
}

// Draw static frame (called once)
void drawFrame() {
  M5.Lcd.fillScreen(C_BG);

  // Top accent bar
  M5.Lcd.fillRect(0, 0, SCR_W, TITLE_H, 0x0010);   // very dark blue-black

  // Title
  M5.Lcd.setTextColor(C_ACCENT, 0x0010);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(8, 5);
  M5.Lcd.print("BEAMNG");

  // Thin right-side separator for gear box
  M5.Lcd.drawFastVLine(GEAR_X - 6, TITLE_H, 68, C_DIM);

  // "km/h" unit label (static)
  M5.Lcd.setTextColor(C_UNIT, C_BG);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(14, 72);
  M5.Lcd.print("km/h");

  // "GEAR" label
  M5.Lcd.setTextColor(C_UNIT, C_BG);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(GEAR_X, GEAR_Y + 2);
  M5.Lcd.print("GEAR");

  // Divider before RPM bar
  M5.Lcd.drawFastHLine(0, RPM_BAR_Y - 6, SCR_W, C_DIVIDER);

  // RPM scale labels
  M5.Lcd.setTextColor(C_DIM, C_BG);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(RPM_BAR_X,          RPM_BAR_Y + RPM_BAR_H + 4);
  M5.Lcd.print("0");
  M5.Lcd.setCursor(RPM_BAR_X + 54,     RPM_BAR_Y + RPM_BAR_H + 4);
  M5.Lcd.print("2k");
  M5.Lcd.setCursor(RPM_BAR_X + 109,    RPM_BAR_Y + RPM_BAR_H + 4);
  M5.Lcd.print("4k");
  M5.Lcd.setCursor(RPM_BAR_X + 164,    RPM_BAR_Y + RPM_BAR_H + 4);
  M5.Lcd.print("6k");
  M5.Lcd.setCursor(RPM_BAR_X + 208,    RPM_BAR_Y + RPM_BAR_H + 4);
  M5.Lcd.print("8k");

  // "RPM" label (top-left of bar)
  M5.Lcd.setTextColor(C_ACCENT, C_BG);
  M5.Lcd.setCursor(RPM_BAR_X, RPM_BAR_Y - 14);
  M5.Lcd.print("RPM");

  // Brake indicator label (top-right of bar area)
  M5.Lcd.setTextColor(C_DIM, C_BG);
  M5.Lcd.setCursor(SCR_W - 40, RPM_BAR_Y - 14);
  M5.Lcd.print("BRK");
}

// Draw segmented RPM bar
void drawRPMBar(int rpm) {
  const int SEG_COUNT = 40;
  const int SEG_GAP   = 2;
  int segW = (RPM_BAR_W - (SEG_COUNT - 1) * SEG_GAP) / SEG_COUNT;
  int filled = (int)((float)rpm / MAX_RPM * SEG_COUNT);
  if (filled > SEG_COUNT) filled = SEG_COUNT;

  for (int i = 0; i < SEG_COUNT; i++) {
    int x = RPM_BAR_X + i * (segW + SEG_GAP);
    float ratio = (float)i / SEG_COUNT;
    uint16_t col = (i < filled) ? rpmColor(ratio) : C_DIM;
    M5.Lcd.fillRect(x, RPM_BAR_Y, segW, RPM_BAR_H, col);
  }
}

// Draw speed number
void drawSpeed(int spd) {
  // Erase old value
  M5.Lcd.fillRect(0, SPEED_Y, GEAR_X - 10, 52, C_BG);

  // Large speed number
  M5.Lcd.setTextColor(C_SPEED, C_BG);
  M5.Lcd.setTextSize(5);

  char buf[8];
  snprintf(buf, sizeof(buf), "%3d", spd);

  M5.Lcd.setCursor(14, SPEED_Y);
  M5.Lcd.print(buf);
}

// Draw gear indicator
void drawGear(const char* gear) {
  // Background box
  uint16_t boxCol = (strcmp(gear, "N") == 0) ? 0x0208 :
                    (strcmp(gear, "R") == 0) ? 0x4000 : 0x0010;
  M5.Lcd.fillRoundRect(GEAR_X - 2, GEAR_Y + 14, 44, 40, 6, boxCol);

  // Gear character(s)
  M5.Lcd.setTextColor(C_GEAR, boxCol);
  M5.Lcd.setTextSize(4);

  int cx = GEAR_X - 2 + (44 - strlen(gear) * 6 * 4) / 2;
  M5.Lcd.setCursor(cx, GEAR_Y + 19);
  M5.Lcd.print(gear);
}

// Draw RPM numeric readout
void drawRPMNumber(int rpm) {
  M5.Lcd.fillRect(55, RPM_BAR_Y - 16, 90, 12, C_BG);
  M5.Lcd.setTextColor(C_UNIT, C_BG);
  M5.Lcd.setTextSize(1);
  char buf[16];
  snprintf(buf, sizeof(buf), "%d rpm", rpm);
  M5.Lcd.setCursor(55, RPM_BAR_Y - 14);
  M5.Lcd.print(buf);
}

// Brake indicator
void drawBrake(bool braking) {
  uint16_t col = braking ? 0xF800 : C_DIM;
  M5.Lcd.fillRoundRect(SCR_W - 36, RPM_BAR_Y - 15, 28, 11, 3, col);
}

// WiFi connecting splash
void drawConnecting() {
  M5.Lcd.fillScreen(C_BG);
  M5.Lcd.setTextColor(C_ACCENT, C_BG);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setCursor(28, 40);
  M5.Lcd.print("Connecting");
  M5.Lcd.setTextSize(1);
  M5.Lcd.setTextColor(C_UNIT, C_BG);
  M5.Lcd.setCursor(50, 70);
  M5.Lcd.print(WIFI_SSID);
}

void drawConnected() {
  M5.Lcd.fillScreen(C_BG);
  M5.Lcd.setTextColor(C_RPM_LO, C_BG);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setCursor(44, 30);
  M5.Lcd.print("Connected");

  M5.Lcd.setTextColor(C_UNIT, C_BG);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(20, 58);
  M5.Lcd.print("IP: ");
  M5.Lcd.setTextColor(C_ACCENT, C_BG);
  M5.Lcd.print(WiFi.localIP().toString());

  M5.Lcd.setTextColor(C_DIM, C_BG);
  M5.Lcd.setCursor(14, 80);
  M5.Lcd.print("Waiting for BeamNG...");

  M5.Lcd.setTextColor(C_DIM, C_BG);
  M5.Lcd.setCursor(14, 100);
  M5.Lcd.print("UDP port: ");
  M5.Lcd.setTextColor(C_UNIT, C_BG);
  M5.Lcd.print(UDP_PORT);

  delay(3000);
}

// Setup
void setup() {
  M5.begin();
  M5.Lcd.setRotation(3);             // Landscape, USB on right
  M5.Lcd.setBrightness(80);
  Serial.begin(115200);

  drawConnecting();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    M5.Lcd.setTextColor(C_ACCENT, C_BG);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setCursor(90 + dots * 8, 90);
    M5.Lcd.print(".");
    dots = (dots + 1) % 10;
  }

  udp.begin(UDP_PORT);
  drawConnected();
  drawFrame();

  // Initial values
  drawSpeed(0);
  drawGear("N");
  drawRPMBar(0);
  drawRPMNumber(0);
  drawBrake(false);
}

// Loop
void loop() {
  M5.update();

  // Receive UDP packet
  int pktSize = udp.parsePacket();
  if (pktSize > 0 && pktSize < (int)sizeof(packetBuf)) {
    int len = udp.read(packetBuf, sizeof(packetBuf) - 1);
    if (len > 0) {
      packetBuf[len] = '\0';
      // Expected format: "RPM,SPEED,GEAR,THROTTLE,BRAKE\n"
      int  rpm, spd, throttle, brake;
      char gear[4] = {0};
      int parsed = sscanf(packetBuf, "%d,%d,%3[^,],%d,%d",
                          &rpm, &spd, gear, &throttle, &brake);

      if (parsed >= 3) {
        g_rpm      = rpm;
        g_speed    = spd;
        g_throttle = throttle;
        g_brake    = (parsed >= 5) ? brake : 0;
        strncpy(g_gear, gear, sizeof(g_gear) - 1);
      }
    }
  }

  // Redraw only changed elements to avoid flicker
  if (g_rpm != g_prev_rpm) {
    drawRPMBar(g_rpm);
    drawRPMNumber(g_rpm);
    g_prev_rpm = g_rpm;
  }

  if (g_speed != g_prev_speed) {
    drawSpeed(g_speed);
    g_prev_speed = g_speed;
  }

  if (strcmp(g_gear, g_prev_gear) != 0) {
    drawGear(g_gear);
    strncpy(g_prev_gear, g_gear, sizeof(g_prev_gear));
  }

  bool braking = g_brake > 5;
  if (braking != g_prev_brake) {
    drawBrake(braking);
    g_prev_brake = braking;
  }

  // Button A (M5 button): toggle display brightness
  if (M5.BtnA.wasPressed()) {
    static int bright = 80;
    bright = (bright == 80) ? 40 : 80;
    M5.Lcd.setBrightness(bright);
  }

  delay(10);  // ~100 Hz loop cap
}
