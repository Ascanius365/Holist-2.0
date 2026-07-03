from javascript import require
vec3 = require("vec3")
from datetime import datetime, timedelta

top_blocks = 30


def observe(bot, mcData, chat_history_buffer, chat_history, event_history):
    observation_data = {}


    # Append entities
    entities = bot.entities
    nearby_players = []

    # Go through all entities
    for entity_id in entities:
        entity = entities[entity_id]

        # Check if not self
        if entity and hasattr(entity, 'username') and entity.username:
            if entity.username != bot.username:
                nearby_players.append(entity.username)

    if nearby_players:
        names_list = ", ".join(nearby_players)
        observation_data["entities"] = (f"   - Nearby players/bots: {names_list}.")


    # Append tool feedback
    clean_chat = [str(msg) for msg in chat_history_buffer if msg is not None]
    if clean_chat:
        observation_data["Tool feedback"] = ("   - Recent tool feedback: " + " | ".join(clean_chat))
    else:
        observation_data["Tool feedback"] = "None"


    # Append chat history
    if chat_history:
        observation_data["Chat history"] = (f"   - Chat history: {chat_history}")


    # Hunger status
    food = int(bot.food)
    if food <= 18:
        observation_data["food"] = (f"   - Food: {food}/20 (If < 18, you won't heal. Eat something!)")

    # Inventory
    items = bot.inventory.items()
    if items:
        observation_data["inventory"] = {
            str(i.name): int(i.count) for i in items
        }

    # Scan nearby blocks
    """Scannt jeden Block in einem Kubus um den Bot."""
    x_radius = 16
    y_radius = 2
    z_radius = 16
    block_stats = {}

    # Position of the bot
    pos = bot.entity.position

    # Loop through X, Y and Z coordinates
    for x in range(int(pos.x) - x_radius, int(pos.x) + x_radius + 1):
        for y in range(int(pos.y) - y_radius, int(pos.y) + y_radius + 1):
            for z in range(int(pos.z) - z_radius, int(pos.z) + z_radius + 1):
                block = bot.blockAt(vec3(x, y, z))

                # 1. Sicherstellen, dass überhaupt ein Block existiert und es keine Luft ist
                if block and block.name != 'air':

                    # Spezialfall: Weizen nach Reifegrad aufteilen
                    if block.name == 'wheat':
                        status = "ripe" if block.metadata == 7 else "not ripe"
                        stat_key = f"{block.name} ({status})"

                        if stat_key in block_stats:
                            block_stats[stat_key] += 1
                        else:
                            block_stats[stat_key] = 1

                    # Normaler Fall: Alle anderen Blöcke
                    else:
                        if block.name in block_stats:
                            block_stats[block.name] += 1
                        else:
                            block_stats[block.name] = 1


    if block_stats:
        top_blocks = sorted(block_stats.items(), key=lambda item: item[1], reverse=True)[:30]
        observation_data["surroundings"] = {str(k): int(v) for k, v in top_blocks}


    # Add Minecraft Time
    current_time = get_minecraft_time(bot)
    observation_data["time"] = f"{current_time}"
    #print(observation_data["time"])


    # Append event history
    if event_history:
        observation_data["Event history"] = event_history

    # Append the position
    pos = bot.entity.position
    block = bot.blockAt(pos)
    biomeId = block.biome.id
    biome_name = mcData.biomes[biomeId].name

    observation_data["Current position"] = (f"   - You are at the coordinates "
                                            f"x: {int(pos.x)}, y: {int(pos.y)}, z: {int(pos.z)}"
                                            f"in biome '{biome_name}'")
    #print(observation_data["Current position"])


    # Armor slots
    helmet = bot.inventory.slots[5].name if bot.inventory.slots[5] else "nothing"
    chestplate = bot.inventory.slots[6].name if bot.inventory.slots[6] else "nothing"
    leggings = bot.inventory.slots[7].name if bot.inventory.slots[7] else "nothing"
    boots = bot.inventory.slots[8].name if bot.inventory.slots[8] else "nothing"

    observation_data["Armor slots"] = (f"   - The bot is currently wearing {helmet}, {chestplate}, {leggings}, {boots}.")


    return observation_data


def get_minecraft_time(bot):
    """
    Translates Minecraft ticks into a readable time and date.

    time_of_day (int): Ticks in the current day (0-23999)
    total_world_age (int): Total ticks in the world
    """

    time_of_day = bot.time.timeOfDay

    time = time_of_day % 24000

    if 0 <= time < 1000:
        phase = "Sunrise (Early Morning)"
    elif 1000 <= time < 6000:
        phase = "Morning"
    elif 6000 <= time < 9000:
        phase = "Noon (Midday)"
    elif 9000 <= time < 12000:
        phase = "Afternoon"
    elif 12000 <= time < 13000:
        phase = "Sunset (Dusk)"
    elif 13000 <= time < 18000:
        phase = "Night"
    elif 18000 <= time < 22000:
        phase = "Midnight"
    else:
        phase = "Late Night (Pre-Dawn)"

    day_count = bot.time.day + 1

    # 1. Uhrzeit berechnen (Start um 06:00 Uhr bei 0 Ticks)
    hours = int((time_of_day / 1000 + 6) % 24)
    minutes = int((time_of_day % 1000) * 60 / 1000)
    time_string = f"{hours:02d}:{minutes:02d}"

    # 3. Wochentag simulieren
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekdays[(day_count - 1) % 7]

    # 4. Datum simulieren (Start am 01.01.2026)
    start_date = datetime(2026, 1, 1)
    current_date = start_date + timedelta(days=day_count - 1)
    date_string = current_date.strftime("%Y-%m-%d")

    # Ergebnis im Format: "2026-01-07 Wednesday 16:58:00"
    full_format = f"{date_string} {weekday} {time_string}:00 {phase}"

    return full_format


def scan_blocks(bot):
    # Scan nearby blocks for structural awareness
    x_radius = 5  # Kleinerer Radius für präzise Koordinaten, sonst wird der Prompt zu lang
    y_radius = 5
    z_radius = 5
    structural_data = []

    pos = bot.entity.position

    for x in range(int(pos.x) - x_radius, int(pos.x) + x_radius + 1):
        for y in range(int(pos.y) - y_radius, int(pos.y) + y_radius + 1):
            for z in range(int(pos.z) - z_radius, int(pos.z) + z_radius + 1):
                block = bot.blockAt(vec3(x, y, z))

                # Luft ignorieren, aber Strukturblöcke erfassen
                if block and block.name != 'air' and block.name != 'cave_air':
                    # Speichere relative Koordinaten (optional), damit der Bot
                    # ein Muster erkennt, egal wo er steht. Hier nutzen wir absolute:
                    structural_data.append(f"{block.name} at ({x},{y},{z})")

    if structural_data:
        # Wir limitieren die Liste, damit das LLM nicht überfordert wird
        # Der Fokus liegt auf den nächsten Blöcken
        obs_text = " | ".join(structural_data[:1000])
        msg = (
            f"Precise block structure detected: {obs_text}. "
            "Use these coordinates to align your building plan."
        )
        #print(msg)
        return msg
    else:
        msg = "No structural blocks nearby to align with."
        #print(msg)
        return msg

