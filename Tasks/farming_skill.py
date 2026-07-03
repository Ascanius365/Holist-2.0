from javascript import require
Vec3       = require('vec3').Vec3
pathfinder = require('mineflayer-pathfinder')

from Tasks.movement import pathfind_to_goal, collect_drops, findClosestBlock
from Tasks.inventory import wieldItem
import time


farming_blocks = ["Wheat Crops"]
farming_items  = ["Wheat"]
farming_seeds  = ["Wheat Seeds"]


# Main loop
# - plant new crops
# - harvest ripe crops
# - deposit in chest

def findHarvestable(bot, r):
    return findClosestBlock(bot, farming_blocks, r, y_radius=1, metadata=7)

def findSoil(bot, r):
    return findClosestBlock(bot, "Farmland", r, y_radius=1, spaceabove=True)

def findBlock(bot, mcData, item):

    blockType = mcData.blocksByName[item]
    if not blockType:
        msg = ("There is no block named: " + item)
        #print(msg)

    # Get Block-object
    block = bot.findBlock({
        "matching": blockType.id,
        "maxDistance": 30
    })

    if not block:
        return None

    return block

async def doFarming(bot, mcData, count):

    if count == 1:
        count = 10

    count = 1

    up = Vec3(0, 1, 0)

    harvested = 0

    # Testing if its ripe
    b = findBlock(bot, mcData, "wheat")
    if not b:
        return "The wheat is not yet ready to be harvested."

    for i in range(count):
        #b = findBlock(bot, mcData, "wheat")
        b = findHarvestable(bot, 30)
        if b:
            await pathfind_to_goal(bot, b, "Wheat Crops")
            try:
                bot.dig(b)
            except Exception as e:
                #print("error while harvesting:",e)
                return f"error while harvesting: {e}"
            #time.sleep(0.2)
            harvested += 1
        else:
            break

    # Collect all drops
    await collect_drops(bot, mcData, "...")

    wieldItem(bot, mcData, "wheat_seeds")
    for i in range(1):
        b = findSoil(bot, 30)
        if b:
            await pathfind_to_goal(bot, b, "Farmland")
            try:
                bot.placeBlock(b,up)
            except Exception as e:
                #print("error while planting:",e)
                return f"error while planting: {e}"
        else:
            break

    #print(f"You have successfully harvested {harvested}x Wheed.")
    return f"You have successfully harvested {harvested}x Wheed."