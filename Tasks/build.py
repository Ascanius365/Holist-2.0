import math, time
from utils.vec3_conversion import vec3_to_str
from javascript import require
from Tasks.movement import goToCoordinates

minecraft_data = require('minecraft-data')
pathfinder = require('mineflayer-pathfinder')

mcdata = minecraft_data("1.21.1")
vec3 = require("vec3")


def get_block_id(block_name):
    block_id = None
    block = mcdata.blocksByName[block_name]
    if block is not None:
        block_id = block.id
    return block_id


def get_empty_block_names():
    return ["air", "water", "lava", "grass", "short_grass", "tall_grass", "snow", "dead_bush", "fern"]


def get_entity_position(entity):
    """Return the (x, y, z) position of a given entity; call with get_entity_position(entity), where entity is a valid entity object."""
    pos = None
    if entity is not None:
        pos = entity.position
    return pos


def get_an_item_in_inventory(bot, item_name, exclude=None):
    """Return an item matching 'item_name' from the agent's inventory if available; call with get_an_item_in_inventory(agent, item_name)."""
    if exclude is None or not isinstance(exclude, list):
        exclude = []
    items = list(filter(lambda item: item_name in item.name and all(name not in item.name for name in exclude),
                        bot.inventory.items()))
    item = items[0] if len(items) > 0 else None
    return item


def get_cant_build_off_block_names():
    return ["bed", "_table", "furnace", "chest"]


def get_display_name_of_block(block_name):
    display_name = get_block_display_name(get_block_id(block_name))
    display_name = display_name if display_name is not None else block_name
    return display_name


def get_block_display_name(block_id):
    block_display_name = None
    block = mcdata.blocks[block_id]
    if block is not None:
        block_display_name = block.displayName
    return block_display_name


# ============ ÄNDERUNG 1: break_block_at ist jetzt ASYNC ============
async def break_block_at(bot, block):
    """Break the block located at coordinates (x, y, z); call with await break_block_at(agent, x, y, z)."""

    if block is None or block.name in get_empty_block_names() or block.name == "water" or block.name == "lava":
        msg = f"Block at is empty or cannot break ({block.name if block else 'None'})."
        #print(f"⏭️  {msg}")
        return False, msg

    # ============ ÄNDERUNG 5: Grabe den Block ============
    try:
        #print(f"🔨 Breaking {block.displayName} ...")
        bot.dig(block, True)
        time.sleep(0.5)  # Warte kurz bis Block vollständig weg ist
        success_msg = f"I broke {block.displayName}."
        #print(f"✅ {success_msg}")
        return True, success_msg
    except Exception as e:
        msg = f"Error breaking block: {str(e)}"
        #print(f"❌ {msg}")
        return False, msg


async def place_block(bot, block_name, x, y, z, place_on='bottom', dont_cheat=False):
    """Place a 'block_name' at (x, y, z), optionally aligning with 'place_on' surface; 'dont_cheat' controls whether block must come from inventory; call with place_block(agent, block_name, x, y, z, place_on, dont_cheat)."""

    #print(f"🔨 [DEBUG] place_block called: block={block_name}, pos=({x}, {y}, {z}), place_on={place_on}, dont_cheat={dont_cheat}")

    if get_block_id(block_name) is None and block_name != 'air':
        msg = f"{block_name} is invalid block name."
        #print(f"❌ {msg}")
        return msg

    target_dest = [math.floor(x), math.floor(y), math.floor(z)]
    #print(f"🎯 Target destination: {target_dest}")

    if block_name in get_empty_block_names():
        #print(f"📦 {block_name} is empty block, breaking instead")
        success, break_msg = await break_block_at(bot, *target_dest)
        if not success:
            return break_msg

    if bot.modes is not None and bot.modes.isOn('cheat') and not dont_cheat:
        #print("✅ Using cheat mode")
        if bot.restrict_to_inventory:
            block = get_an_item_in_inventory(bot, block_name)
            if block is None:
                msg = f"Cheat mode but item {block_name} not in inventory."
                #print(f"❌ {msg}")
                return msg

        face = "east"
        if place_on == "north":
            face = "south"
        elif place_on == "south":
            face = "north"
        elif place_on == "east":
            face = "west"

        if "torch" in block_name and place_on != "bottom":
            block_name = block_name.replace('torch', 'wall_torch')
            if place_on != "side" and place_on != "top":
                block_name += "[facing=%s]" % face

        if "botton" in block_name or block_name == "lever":
            if place_on == "top":
                block_name += "[face=ceiling]"
            elif place_on == "bottom":
                block_name += "[face=floor]"
            else:
                block_name += "[facing=%s]" % face

        if block_name == "ladder" or block_name == "repeater" or block_name == "comparator":
            block_name += "[facing=%s]" % face

        if "stairs" in block_name:
            block_name += "[facing=%s]" % face

        msg = "/setblock %d %d %d %s" % (math.floor(x), math.floor(y), math.floor(z), block_name)
        #print(f"💬 Executing: {msg}")
        bot.chat(msg)

        if "door" in block_name:
            msg = "/setblock %d %d %d %s [half=upper]" % (math.floor(x), math.floor(y + 1), math.floor(z), block_name)
            print(f"💬 Executing door upper: {msg}")
            bot.chat(msg)

        if "bed" in block_name:
            msg = "/setblock %d %d %d %s [part=head]" % (math.floor(x), math.floor(y), math.floor(z - 1), block_name)
            print(f"💬 Executing bed head: {msg}")
            bot.chat(msg)

        msg = f"Successfully placed {get_display_name_of_block(block_name)} at {target_dest}."
        print(f"✅ {msg}")
        return msg

    item_name = block_name
    if item_name == "redstone_wire":
        item_name = "redstone"

    block = get_an_item_in_inventory(bot, item_name)

    print(f"📦 Inventory check for '{item_name}': {block is not None}")
    if block is not None:
        print(f"   Found: {block.name} (count: {block.count})")

    if block is None:
        msg = f"Have no {block_name} to place in inventory."
        print(f"❌ {msg}")
        return msg

    target_block = bot.blockAt(vec3.Vec3(*target_dest))
    print(f"📍 Target block: {target_block.name if target_block else 'None'}")

    if target_block is not None:
        if target_block.name == block_name:
            msg = f"Target already has {block_name}."
            print(f"⏭️  {msg}")
            return msg
        if target_block.name not in get_empty_block_names():

            # ============ ÄNDERUNG 6: Nutze await für async break_block_at ============

            # Gehe zum Block wenn zu weit weg
            agent_pos = get_entity_position(bot.entity)
            if agent_pos is not None and agent_pos.distanceTo(target_block.position) > 4.5:
                print(f"📍 Block is too far ({agent_pos.distanceTo(target_block.position):.2f} blocks), moving closer...")
                move_msg = await goToCoordinates(bot, target_block.position.x,
                                                 target_block.position.y, target_block.position.z)
                print(f"Movement result: {move_msg}")
                if "did not reach" in move_msg.lower():
                    print(f"❌ Could not reach block to break it.")
                    return False, move_msg
                print(f"✅ Reached block position")

            print(f"🔨 Breaking existing block: {target_block.name}")

            success, break_msg = await break_block_at(bot, target_block)
            if not success:
                return break_msg
            time.sleep(0.2)

    build_off_block, face_vec = None, None
    dir_map = {
        "top": [0, 1, 0], "bottom": [0, -1, 0],
        "north": [0, 0, -1], "south": [0, 0, 1],
        "east": [1, 0, 0], "west": [-1, 0, 0],
    }

    dirs = []
    if place_on == "side":
        dirs.extend([dir_map["north"], dir_map["south"], dir_map["east"], dir_map["west"]])
    elif place_on in dir_map.keys():
        dirs.append(dir_map[place_on])
    else:
        dirs.append(dir_map["bottom"])

    for d in dir_map.values():
        if d not in dirs:
            dirs.append(d)

    print(f"🔍 Searching for build-on block in {len(dirs)} directions...")
    for i, d in enumerate(dirs):
        check_pos = [target_dest[0] + d[0], target_dest[1] + d[1], target_dest[2] + d[2]]
        b = bot.blockAt(vec3.Vec3(*check_pos))
        block_info = f"{b.name}" if b else "None"
        print(f"   [{i}] Direction {d}: {block_info}")

        if b is not None and b.name not in get_empty_block_names() and all(
                [n not in b.name for n in get_cant_build_off_block_names()]):
            build_off_block = b
            face_vec = [-d[0], -d[1], -d[2]]
            print(f"   ✅ Found build-on block: {b.name}")
            break

    if build_off_block is None:
        msg = (f"Can't place {block_name} at {target_dest}. Nothing to place on."
               f"You can only place a block directly next to an existing block.")
        print(f"❌ {msg}")
        return msg

    agent_pos = get_entity_position(bot.entity)
    print(f"🤖 Agent position: {agent_pos}")

    if agent_pos is not None:
        distance = agent_pos.distanceTo(vec3.Vec3(*target_dest))
        print(f"📏 Distance to target: {distance:.2f} blocks")

        pos_above = agent_pos.plus(vec3.Vec3(0, 1, 0))
        dont_move_for = ['torch', 'redstone_torch', 'redstone_wire', 'lever', 'button', 'rail', 'detector_rail',
                         'powered_rail', 'activator_rail', 'tripwire_hook', 'tripwire', 'water_bucket']

        if block_name not in dont_move_for and (
                agent_pos.distanceTo(target_block.position) < 1 or pos_above.distanceTo(target_block.position) < 1):
            print(f"⚠️  Too close to target, moving away...")
            goal = pathfinder.goals.GoalNear(target_dest[0], target_dest[1], target_dest[2], 2)
            inverted_goal = pathfinder.goals.GoalInvert(goal)
            bot.pathfinder.setMovements(pathfinder.Movements(bot))
            bot.pathfinder.setGoal(inverted_goal)
            time.sleep(0.1)
            while bot.pathfinder.isMoving():
                time.sleep(0.2)
            print(f"✅ Moved away from target")

    agent_pos = get_entity_position(bot.entity)
    if agent_pos is not None and agent_pos.distanceTo(vec3.Vec3(*target_dest)) > 4.5:
        print(f"📍 Too far from target, moving closer...")
        move_msg = await goToCoordinates(bot, None, target_dest[0], target_dest[1], target_dest[2])
        print(f"Movement result: {move_msg}")
        if "did not reach" in move_msg.lower():
            return move_msg

    print(f"🎯 Placing block...")
    print(f"   Block: {block.name}")
    print(f"   Build on: {build_off_block.name}")
    print(f"   Face vector: {face_vec}")

    try:
        bot.equip(block, 'hand')
        bot.lookAt(build_off_block.position)
        bot.placeBlock(build_off_block, vec3.Vec3(*face_vec))
        msg = f"Successfully placed {get_display_name_of_block(block_name)} at {target_dest}."
        bot.chat(msg)
        print(f"✅ {msg}")
        return msg
    except Exception as e:
        msg = f"Error placing block: {str(e)}"
        print(f"❌ {msg}")
        return msg