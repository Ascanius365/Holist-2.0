from javascript import require, On
import asyncio
from Tasks.inventory import wieldItem
from Tasks.movement import collect_drops

async def attackMob(bot, mcData, item):

    name = item

    try:
        # Equpt weapon and armor
        weapon = "iron_sword"
        msg = wieldItem(bot, mcData, weapon)

        if msg:
            #print(msg)
            return msg

        bot.armorManager.equipAll()

        best_entity = None
        # float('inf') entspricht dem JavaScript Number.MAX_VALUE
        best_distance = float('inf')

        # Über die Bridge iterieren wir am stabilsten über die IDs (Keys) von bot.entities
        for entity_id in bot.entities:
            entity = bot.entities[entity_id]

            # Überspringe den Bot selbst (ID-Vergleich ist über die Runtime-Grenze hinweg sicherer)
            if entity.id == bot.entity.id:
                continue

            # Filtern nach dem Minecraft-Namen des Mobs
            if entity.name != "pig":
                continue

            # distanceSquared nutzt die native Mineflayer/Vec3-Methode
            dist = bot.entity.position.distanceSquared(entity.position)
            if dist < best_distance:
                best_entity = entity
                best_distance = dist

        if best_entity == None or best_distance > 50:
            msg = f"{name} wasn\'t found."
            #print(msg)
            return msg
        else:
            bot.pvp.attack(best_entity)
            #print(f"Attacking {name}.")

        """
        player = bot.players[name]
        if player == None:
            msg = f"Player {name} wasn\'t found."
            print(msg)
            return  msg
        else:
            await bot.pvp.attack(player.entity)
            print(f"Attacking {name}.")"""

        await collect_drops(bot, mcData, name)

        msg = f"The {name} was successfully killed!"
        #print(msg)
        return msg

    except Exception as e:
        print(e)