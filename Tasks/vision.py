import asyncio
import base64
import io
import json
import os
import re
import time
from typing import Dict, Any
from pathlib import Path

from PIL import Image
from openai import OpenAI
from playwright.async_api import async_playwright
from simple_chalk import chalk


class VisionManager:
    def __init__(self, port, bot_name):
        self.url = f"http://localhost:{port}"
        self.port = port

        self.bot_name = bot_name
        self.bot_dir = f"bots/{bot_name}/memory"
        self.sessions_file = f"{self.bot_dir}/sessions.jsonl"

    # ============ ÄNDERUNG 2: Neue Funktion zum Parsing der Vision-Description ============
    def parse_vision_to_spatial_facts(self, description: str, bot_pos: Dict, observation_time: str) -> Dict[str, Any]:
        """
        Parst die Vision-Description und extrahiert spatial facts.
        Findet Koordinaten im Format [X, Y, Z] und erstellt structured spatial facts.

        Returns:
            Dict mit 'spatial' key und List von facts
        """
        spatial_facts = []

        # ============ ÄNDERUNG 2a: Finde alle [X, Y, Z] Koordinaten ============
        # Pattern: [123.5, 456, -789] oder ähnlich
        coordinate_pattern = r'\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]'

        coordinates = re.finditer(coordinate_pattern, description)

        for match in coordinates:
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))

            # Extrahiere die Beschreibung vor/nach der Koordinate
            # Suche nach dem nächsten Punkt oder Komma
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(description), match.end() + 50)
            context = description[start_pos:end_pos].strip()

            # ============ ÄNDERUNG 2b: Erstelle strukturiertes Insight ============
            spatial_fact = {
                "key": self._extract_location_key(context, x, y, z),
                "value": f"x: {x}, y: {y}, z: {z}",
                "date": observation_time,
                "message_id": f"vision_{int(time.time())}"
            }
            spatial_facts.append(spatial_fact)

        # ============ ÄNDERUNG 2c: Wenn keine Koordinaten gefunden, extrahiere descriptive facts ============
        if not spatial_facts:
            # Parse textuelle Beschreibungen
            lines = description.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Teile in key:value
                    if ':' in line:
                        key, value = line.split(':', 1)
                        spatial_fact = {
                            "key": key.strip(),
                            "value": value.strip(),
                            "date": observation_time,
                            "message_id": f"vision_{int(time.time())}"
                        }
                        spatial_facts.append(spatial_fact)

        return {
            "spatial": spatial_facts
        }

    def _extract_location_key(self, context: str, x: float, y: float, z: float) -> str:
        """Extrahiert einen aussagekräftigen Key aus dem Context"""
        # Entferne Koordinaten aus dem Context
        clean_context = re.sub(r'\[.*?\]', '', context).strip()

        # Erste 50 Zeichen als Key
        key = clean_context[:50] if clean_context else f"Location [{x}, {y}, {z}]"
        return key.rstrip('.,;:')

    async def get_birdseye_screenshot(self, bot, observation, output_path="birdseye_view.png"):

        print("Make foto")

        output_path = f"bots/{self.bot_name}/birdseye_view.png"
        """
        Macht einen Draufsicht-Screenshot des Viewers.
        Optimiert für Prismarine-Viewer mit MapControls/OrbitControls.
        """
        try:
            print(f"{self.bot_name}: 🦅 Vision: Starte Vogelperspektive auf {self.url}...")

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                # Quadratisches Sichtfeld für Karten-Optik
                await page.set_viewport_size({"width": 1024, "height": 1024})

                #print(f"🌍 Lade Viewer auf {self.url}...")
                await page.goto(self.url, wait_until="networkidle", timeout=20000)

                #print("⏳ Warte auf Welt-Rendering (6 Sekunden)...")
                await asyncio.sleep(6)

                #print("🎥 Setze Kamera-Position via JavaScript...")

                setup_script = """
                async () => {
                    const viewer = window.viewer || window.botViewer || (window.app && window.app.viewer);

                    if (!viewer) {
                        console.error("Viewer-Instanz nicht gefunden!");
                        return { success: false, error: "Viewer not found" };
                    }

                    try {
                        const targetPos = viewer.camera.position.clone();
                        viewer.camera.position.set(targetPos.x, targetPos.y + 60, targetPos.z);
                        viewer.camera.lookAt(targetPos.x, targetPos.y, targetPos.z);

                        if (viewer.controls) {
                            if (viewer.controls.target) {
                                viewer.controls.target.set(targetPos.x, targetPos.y, targetPos.z);
                            }
                            viewer.controls.update();
                        }

                        if (typeof viewer.update === 'function') viewer.update();

                        return { success: true };
                    } catch (e) {
                        return { success: false, error: e.message };
                    }
                }
                """

                result = await page.evaluate(setup_script)
                if not result or not result.get("success"):
                    error_msg = result.get("error") if result else "Unknown"
                    print(f"{self.bot_name}: ⚠️ JS-Steuerung fehlgeschlagen ({error_msg}), versuche Maus-Fallback...")

                    await page.click("canvas")
                    for _ in range(15):
                        await page.mouse.wheel(0, 1000)
                        await asyncio.sleep(0.1)

                await asyncio.sleep(2)

                #print("📷 Mache Screenshot...")
                await page.screenshot(path=output_path)

                with Image.open(output_path) as img:
                    img.thumbnail((1024, 1024))
                    img.save(output_path)

                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                await browser.close()

        except Exception as e:
            print(f"{self.bot_name}: ❌ Screenshot Fehler: {e}")
            return None

        # Extrahiere die Koordinaten für den Prompt
        bot_pos = bot.entity.position

        curr_x = round(bot_pos['x'], 1)
        curr_y = round(bot_pos['y'], 1)
        curr_z = round(bot_pos['z'], 1)

        view_distance_approx = 80

        # --- TEIL 2: VISION-ANALYSE UND SPATIAL FACTS EXTRACTION ---
        try:

            from dotenv import load_dotenv
            env_path = Path(__file__).parent / '.env'
            success = load_dotenv(dotenv_path=env_path)

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                #api_key=os.environ.get("OPENAI_API_KEY"),
                api_key=os.getenv("OPENROUTER_API_KEY")
            )

            system_prompt = (
                f"You are a precise Minecraft Navigation and Vision Expert. The bot is located exactly "
                f"at the center of this image at the following world coordinates: X={curr_x}, Y={curr_y}, Z={curr_z}.\n\n"
                "Your task is to analyze the image and calculate the world coordinates for all notable "
                "objects (structures, mobs, players, resources) based on the following spatial mapping:\n"
                "- RIGHT of the center: Increasing X direction (+X)\n"
                "- LEFT of the center: Decreasing X direction (-X)\n"
                "- UP from the center (Top of image): Decreasing Z direction (-Z) (North)\n"
                "- DOWN from the center (Bottom of image): Increasing Z direction (+Z) (South)\n"
                "- Y-Axis (Altitude): Estimate based on terrain features, shadows, and block layers.\n\n"
                f"SCALE INFORMATION:\n"
                f"- The image covers an area of approximately {view_distance_approx} x {view_distance_approx} blocks.\n"
                f"- This means from the center to any edge (top, bottom, left, right) is about {view_distance_approx // 2} blocks.\n\n"
                "INSTRUCTIONS:\n"
                "1. Identify all significant landmarks and entities.\n"
                "2. For every identified object, provide estimated world coordinates in the format: [X, Y, Z].\n"
                "3. Be as accurate as possible with block-distance estimation.\n"
                "4. Format: 'Object description [X, Y, Z]'"
            )

            user_content = [
                {
                    "type": "text",
                    "text": "Describe the surroundings in this picture with coordinates."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = client.chat.completions.create(
                model="qwen/qwen3-vl-235b-a22b-instruct",
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )

            description = response.choices[0].message.content.strip()
            #print(chalk.cyan(f"👁️ Vision-Agent Analyse:\n{description}"))

            # ============ ÄNDERUNG 3: Parse zu einzelnen spatial facts ============
            # Die Funktion gibt nun direkt das strukturierte Dict zurück
            formatted_facts = self.parse_vision_to_facts_list(
                description,
                {"x": curr_x, "y": curr_y, "z": curr_z},
                observation.get("time", "")
            )

            # ============ ÄNDERUNG 4: In jsonl schreiben ============
            sessions_entry = {
                "session_id": f"vision_{int(time.time())}",
                "session_date": observation.get("time", ""),
                "role": "vision",
                "response": formatted_facts  # ← Enthält jetzt die Liste einzelner Fakten!
            }

            names = ["Caleb", "Dylan", "Kelly"]

            for name in names:

                bot_dir = f"bots/{name}/memory"
                sessions_file = f"{bot_dir}/sessions.jsonl"

                # Stelle sicher dass Verzeichnis existiert
                os.makedirs(os.path.dirname(sessions_file), exist_ok=True)

                with open(sessions_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sessions_entry, ensure_ascii=False) + "\n")

                print(chalk.green(f"{name}: ✅ Spatial facts saved in {sessions_file}"))

        except Exception as api_e:
            print(f"❌ API failure: {api_e}")
            return None


    def parse_vision_to_facts_list(self, description, bot_pos, session_date):
        """
        Nimmt den rohen Textblock des Vision-Modells, splittet ihn zeilenweise
        und bereitet ihn für das Episodic_Fact-Format des Langzeitgedächtnisses vor.
        """
        facts_list = []
        lines = description.splitlines()

        for line in lines:
            line = line.strip()

            # Unwichtige Zeilen, Überschriften oder leere Zeilen überspringen
            if not line or line.startswith("Notable objects") or line.startswith("Bot position"):
                continue

            # Jeden einzelnen Fund als eigenen Fakt verpacken
            facts_list.append({
                "key": "Visual_Observation",
                "value": line,
                "date": session_date,  # Reicht das Datum für spätere Suchen durch
                "message_id": "v0"
            })

        # Das fertige Dict im von update_long_term_memory erwarteten Format zurückgeben
        return {
            "Episodic_Fact": facts_list
        }