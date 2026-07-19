import asyncio

from javascript import require, On, off
from simple_chalk import chalk
import time
import traceback
import multiprocessing
import os

from Agent import run_agent_async, fetch_agent_response, start_agent_process, init_queues
from Tasks.build import place_block, mcdata
from Tasks.craft import craft
from Tasks.inventory import Chest, smeltItem, clearFurnace
from Tasks.mine import dig
from Tasks.movement import goToCoordinates
from Tasks.needs import eat
from Tasks.observation import observe, scan_blocks
from Tasks.vision import VisionManager
from Tasks.farming_skill import doFarming
from Tasks.pvp import attackMob


# Import the javascript libraries
mineflayer = require("mineflayer")
pathfinder = require("@miner-org/mineflayer-baritone").loader
pathfinder2 = require("mineflayer-pathfinder").pathfinder
viewer = require('prismarine-viewer')
pvp = require("mineflayer-pvp").plugin
armor_manager = require("mineflayer-armor-manager")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Global bot parameters
server_host = "localhost"
server_port = 3000
reconnect = True
version = "1.21.1"
hideErrors = False


class MCBot:
    def __init__(self, bot_name, viewer_port, embedding_req_queue, embedding_res_queue):
        self.bot_name = bot_name
        # Individual queues and processes for this bot
        self.request_queue, self.response_queue = init_queues()

        self.agent_process = start_agent_process(
            self.request_queue,
            self.response_queue,
            bot_name,
            embed_req_q=embedding_req_queue,
            embed_res_q=embedding_res_queue
        )

        self.mineflayer_pathfinder = pathfinder
        self.pathfinder2 = pathfinder2
        self.mineflayer_pvp = pvp
        self.mineflayer_armor_manager = armor_manager

        self.bot_args = {
            "username": bot_name,
            "host": server_host,
            "port": server_port,
            "version": version,
            "hideErrors": False,
        }
        self.agent_is_busy = False
        self.chat_history_buffer = []
        self.chat_history = []
        self.event_history = []
        self.dropped_items = []

        # HIER DYNAMISCH: Jedes Dorfmitglied bekommt ein eigenes Auge
        self.viewer_port = viewer_port
        self.vision = VisionManager(port=viewer_port, bot_name = bot_name)
        self.last_vision_update = 1000
        self.start_bot()


    def start_bot(self):
        self.bot = mineflayer.createBot(self.bot_args)
        self.mcData = require('minecraft-data')(self.bot.version)

        # ============
        # Pathfinder Settings
        # ============

        self.bot.loadPlugin(self.mineflayer_pathfinder)
        self.bot.loadPlugin(self.pathfinder2)
        self.bot.loadPlugin(self.mineflayer_pvp)
        self.bot.loadPlugin(self.mineflayer_armor_manager)

        # Enable / disable features
        self.bot.ashfinder.config.parkour = False  # Allow parkour jumps
        self.bot.ashfinder.config.breakBlocks = True  # Allow breaking blocks
        self.bot.ashfinder.config.placeBlocks = True  # Allow placing blocks
        self.bot.ashfinder.config.swimming = False  # Allow swimming

        # Set limits
        self.bot.ashfinder.config.maxFallDist = 3  # Max safe fall distance
        self.bot.ashfinder.config.maxWaterDist = 256  # Max water distance

        # Configure blocks
        self.bot.ashfinder.config.disposableBlocks = [
            "dirt",
            "cobblestone",
            "stone",
            "andesite",
        ]
        self.bot.ashfinder.config.blocksToAvoid = ["crafting_table", "chest", "furnace", "obsidian"]
        self.bot.ashfinder.config.thinkTimeout = 30000

        self.start_events()


    # Tags bot username before console messages
    def log(self, message):
        print(f"[{self.bot.username}] {message}")


    async def run_vision_background(self, observation):
        """Läuft parallel, nutzt aber die bereits gesammelten Observations"""
        try:
            # Reiche die bestehende observation direkt an die Vision weiter,
            # anstatt die JS-Bridge erneut parallel mit observe() zu belasten!
            await self.vision.get_birdseye_screenshot(self.bot, observation)
        except Exception as e:
            print(f"⚠️ [Background] Vision Task failed: {e}")


    async def main_agent_loop(self):

        while True:
            try:
                # Gather Observations
                observation = observe(self.bot, self.mcData, self.chat_history_buffer, self.chat_history, self.event_history)

                """
                await self.vision.get_birdseye_screenshot(self.bot, observation)
                # Vision-Check (alle 10 Min / 600 Sek)
                if self.bot_name == "Dylan":
                    current_time = time.time()
                    if current_time - self.last_vision_update >= 600:
                        print("📸 Trigger Vision Snapshot...")
                        # Wir übergeben die frische Observation direkt
                        asyncio.create_task(self.run_vision_background(observation))
                        self.last_vision_update = current_time"""


                # Send to YOUR OWN queue
                run_agent_async(self.request_queue, observation)
                self.chat_history_buffer.clear()

                response = await fetch_agent_response(self.response_queue)

                if response and "action" in response:

                    if len(self.event_history) > 0:
                        self.event_history.pop(0)

                    action = response["action"]
                    item = response.get("item")
                    reason = response.get("reasoning")
                    count = response.get("count")
                    x = response.get("x")
                    y = response.get("y")
                    z = response.get("z")

                    try:
                        chat = response.get("chat_intent")
                        if chat:
                            self.bot.chat(chat)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"⚠️ Chat-Fehler (Schleife läuft weiter): {e}")

                    if action == "chat":
                        self.bot.chat(reason)

                    elif action == "eat":
                        msg = await eat(self.bot, self.mcData, item)
                        self.chat_history_buffer.append(msg)

                    elif action == "putInChest":
                        msg = await self.my_chest.depositOneToChest(self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "takeFromChest":
                        msg = await self.my_chest.withdrawOneFromChest(self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "viewChest":
                        msg = await self.my_chest.viewChest(self.mineflayer_pathfinder)
                        self.chat_history_buffer.append(msg)

                    elif action == "dig":
                        msg = await dig(self.bot, self.mcData, item)
                        self.chat_history_buffer.append(msg)

                    elif action == "craft":
                        msg = await craft(self.bot, self.mcData, self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "smeltItem":
                        msg = await smeltItem(self.bot, self.mcData, self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "clearFurnace":
                        msg = await clearFurnace(self.bot, self.mcData, self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "goToCoordinates":
                        msg = await goToCoordinates(self.bot, x, y, z)
                        self.chat_history_buffer.append(msg)

                    elif action == "placeBlock":
                        msg = await place_block(self.bot, item, x, y, z)
                        self.chat_history_buffer.append(msg)

                    elif action == "scanBlocks":
                        msg = scan_blocks(self.bot)
                        self.chat_history_buffer.append(msg)

                    elif action == "doFarming":
                        msg = await doFarming(self.bot, self.mcData, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "attackMob":
                        msg = await attackMob(self.bot, self.mcData, item)
                        self.chat_history_buffer.append(msg)

            except Exception as e:
                traceback.print_exc()
                print(f"Fehler im Loop: {e}")
                return f"Error: {str(e)}"



    # Attach mineflayer events to bot
    def start_events(self):

        # Login event (Logged in)
        @On(self.bot, "login")
        def login(this):
            self.bot_socket = self.bot._client.socket
            self.log(chalk.green(
                f"Logged in to {self.bot_socket.server if self.bot_socket.server else self.bot_socket._host}"
            ))


        @On(self.bot, "spawn")
        def spawn(this):
            # Sicherstellen, dass der Prozess startet
            self.log(chalk.green("Bot gestartet!"))

            # 2. Den Viewer direkt über das Hauptmodul starten
            try:
                viewer.mineflayer(self.bot, {'port': self.viewer_port, 'firstPerson': False})
                print(f"Auge aktiviert für {self.bot_name} auf http://localhost:{self.viewer_port}")
            except Exception as e:
                print(f"Viewer für {self.bot_name} konnte nicht gestartet werden: {e}")

            self.my_chest = Chest(self, self.mcData, chesttype="Chest")

            if self.my_chest.object:
                print(f"Truhe gefunden bei: {self.my_chest.object.position}")

            # We start the asynchronous main loop
            asyncio.run_coroutine_threadsafe(self.main_agent_loop(), loop)

        # Kicked event (Got kicked from server)
        @On(self.bot, "kicked")
        def kicked(this, reason, loggedIn):
            if loggedIn:
                self.log(chalk.red(f"Kicked whilst trying to connect: {reason}"))

        # Chat event: Triggers on chat message
        @On(self.bot, "messagestr")
        def messagestr(this, message, messagePosition, jsonMsg, sender, verified=None):
            if not sender:
                return
            # 1. NACHRICHT ZUM PUFFER HINZUFÜGEN
            self.chat_history.append(message)
            if len(self.chat_history) > 10:
                self.chat_history.pop(0)
            print("message: " + str(self.chat_history))

        @On(self.bot, "death")
        def death(this):
            self.event_history.append("death")

        # End event (Disconnected from server)
        @On(self.bot, "end")
        def end(this, reason):
            self.log(chalk.red(f"Disconnected: {reason}"))

            # Turn off event listeners
            off(self.bot, "login", login)
            off(self.bot, "kicked", kicked)
            off(self.bot, "end", end)
            off(self.bot, "messagestr", messagestr)


if __name__ == '__main__':

    # Liste deiner simulierten Dorfbewohner
    villagers = ["Caleb", "Dylan", "Kelly"]
    #villagers =["Dylan"]

    bots = []
    base_port = 3001

    ctx = multiprocessing.get_context('spawn')

    # 1. Eine zentrale Queue für alle eingehenden Embedding-Anfragen
    embedding_req_queue = ctx.Queue()

    embedding_res_queues = {name: ctx.Queue() for name in villagers}

    """
    # 3. Den zentralen GPU-Worker starten
    embedding_process = ctx.Process(
        target=central_embedding_worker,
        args=(embedding_req_queue, embedding_res_queues)
    )
    embedding_process.start()"""

    print(f"🏘️ Simuliere Dorfleben mit {len(villagers)} Einwohnern...")

    for i, name in enumerate(villagers):
        # Jeder Bot kriegt einen fortlaufenden Port: 3001, 3002, 3003...
        current_viewer_port = base_port + i

        bot_instance = MCBot(
            bot_name=name,
            viewer_port=current_viewer_port,
            embedding_req_queue=embedding_req_queue,
            embedding_res_queue=embedding_res_queues[name]
        )
        bots.append(bot_instance)

        # 3 Sekunden Atempause für das OS/Modell-Loading pro Worker
        time.sleep(3)

    try:
        # Hält Python wach und verarbeitet alle autonomen Schleifen parallel
        loop.run_forever()
    except KeyboardInterrupt:
        print("\n🏘️ Dorf-Simulation wird beendet...")