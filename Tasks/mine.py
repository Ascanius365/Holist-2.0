from javascript import require
vec3 = require("vec3")
from Tasks.inventory import wieldItem
from Tasks.movement import pathfind_to_goal
import asyncio
from Tasks.movement import collect_drops


needs_diamond_pickaxe = ["obsidian"]
needs_iron_pickaxe = ["gold_ore", "redstone_ore", "diamond_ore", "emerald_ore"]
needs_stone_pickaxe = ["copper_ore", "iron_ore"]
needs_wooden_pickaxe = ["stone", "coal_ore", "cobblestone"]

needs_shovel = ["Dirt", "Gravel", "Sand"]

async def dig(bot, mcData, item):

    #print("Search for " + item)
    blockType = mcData.blocksByName[item]
    if not blockType:
        msg = ("There is no block named: " + item)
        #print(msg)
        return msg

    # Get Block-object
    block = bot.findBlock({
        "matching": blockType.id,
        "maxDistance": 30
    })

    if not block:
        msg = (f"Wanted to dig {item}, but there is nothing!")
        #print(msg)
        return msg

    # Check for the right tool
    if item in needs_shovel:
        msg = wieldItem(bot, mcData, "Stone Shovel")
        if msg:
            return (f"{msg} to dig {item}")

    elif item in needs_diamond_pickaxe:
        msg = wieldItem(bot, mcData, "Diamond Pickaxe")
        if msg:
            return (f"{msg} to dig {item}")
    elif item in needs_iron_pickaxe:
        msg = wieldItem(bot, mcData, "Iron Pickaxe")
        if msg:
            return (f"{msg} to dig {item}")
    elif item in needs_stone_pickaxe:
        msg = wieldItem(bot, mcData, "Stone Pickaxe")
        if msg:
            return (f"{msg} to dig {item}")
    elif item in needs_wooden_pickaxe:
        msg = wieldItem(bot, mcData, "Wooden Pickaxe")
        if msg:
            return (f"{msg} to dig {item}")

    # Move to goal
    await pathfind_to_goal(bot, block, item)


    # ========================================
    # Dig Block
    #========================================

    try:
        # Get Block-object again
        block_to_mine = bot.findBlock({
            "matching": blockType.id,
            "maxDistance": 30
        })

        if not block_to_mine:
            return f"Block {item} disappeared before mining!"

        # dig the block
        #print(f"Beginn to dig {item}.")
        bot.dig(block_to_mine, timeout=15000)
        await asyncio.sleep(1)

        await collect_drops(bot, mcData, item)

        #print(f"Successfully obtained 1x {item}!")
        return f"Successfully obtained 1x {item}!"

    except asyncio.TimeoutError:
        #print(f"Mining timed out for {item}")
        return f"Mining {item} timed out - it may still be in your inventory"

    except Exception as e:
        error_msg = str(e)
        #print(f"Error mining {item}: {error_msg}")

        # Check if it's a timeout error
        if "timeout" in error_msg.lower():
            #print(f"⚠️ Mining {item} took too long.")
            return f"Mining {item} took too long."

        return f"Failed to mine {item}: {error_msg}"
