import time
from javascript import require
vec3 = require("vec3")
import asyncio
from Tasks.movement import pathfind_to_goal


def itemTypeAndName(bot, mcData, item_arg):

    item_type = None
    item_name = "Unknown"

    if isinstance(item_arg, int):
        # The input is an item ID
        item_type = item_arg
        itemObj = mcData.items[item_type]

        if itemObj:
            item_name = itemObj.displayName
        else:
            item_type = None
            item_name = "Unknown ID"

    elif isinstance(item_arg, str):

        try:
            # Input is an item arg
            item_data = mcData.itemsByName[item_arg]
            item_arg = item_data.displayName
        except (KeyError, AttributeError, TypeError):
            pass

        # The input is the item name
        item_name = item_arg

        if bot.inventory.items != []:
            for item in bot.inventory.items():
                if item.displayName == item_name:
                    item_type = item.type
                    return item_type, item_name
            else:
                item_type = None

    elif item_arg.type and item_arg.displayName:
        item_type = item_arg.type
        item_name = item_arg.displayName
    else:
        item_type = None
        item_name = "Unknown"

    return item_type, item_name


def checkInHand(bot, mcData, item_arg):

    if not bot.heldItem:
        return False

    item_type, item_name = itemTypeAndName(bot, mcData, item_arg)

    if bot.heldItem.type == item_type:
        return True
    else:
        return False

def itemInHand(mcData, bot):

    if not bot.heldItem:
        return None, "None"
    return bot.heldItem.type, bot.heldItem.displayName


# Equip an Item into the main hand.
def wieldItem(bot, mcData, item_arg):

    item_name = ""
    msg = ""

    if not item_arg:
        #print("trying to equip item 'None'.")
        return None

    item_type, item_name = itemTypeAndName(bot, mcData, item_arg)

    if item_type == None:
        msg = f'You dont have {item_name} in your inventory'
        #print(msg)
        return msg

    time.sleep(0.25)

    if checkInHand(bot, mcData, item_type):
        #print(f'Already holding {item_name}')
        return None

    #print(f'Equipping {item_name} (type: {item_type})')

    for i in range(0, 5):
        try:
            #print(f'wieldItem() attempt #{i + 1}')
            bot.equip(item_type, "hand")
            break
        except Exception as e:
            #print(f'wieldItem() try #{i + 1}. Error: {e}')
            if checkInHand(bot, mcData, item_type):
                #print(f'Item {item_name} is now in hand despite exception')
                return None
            time.sleep(0.5)

    time.sleep(0.25)

    if checkInHand(bot, mcData, item_type):
        #print(f'Successfully equipped {item_name}')
        return None
    else:
        #print(f'Wielding item {item_name} failed after max retries!')
        return None


#==========================
# Chest use
#==========================

class Chest:

    def __init__(self,pybot,mcData,chesttype="Chest",silent=False):
        self.pybot = pybot
        self.mcData = mcData

        # How we find it depends on the type:
        # Chests are blocks
        if chesttype == "Chest":
            blockType = self.mcData.blocksByName["chest"]
            self.object = self.pybot.bot.findBlock({
                "matching": blockType.id,
                "maxDistance": 100
            })
            self.chestType = chesttype

        if self.object == None:
            if not silent:
                print(f'Cant find any {chesttype} nearby.')
        self.container = None

    def open(self):
        try:
            self.container = self.pybot.bot.openContainer(self.object, timeout=5000)
            if not self.container:
                #print("Can't open chest.")
                return f"You cant open the chest."
            time.sleep(0.5)
            return None
        except Exception as e:
            #print(f"Error opening chest: {e}")
            return f"Failed to open chest: {str(e)}"

    def close(self):
        self.container.close()
        self.container = None

    def spaceAvailable(self):
        if self.open():
            chest_size = self.container.inventoryStart
            empty = chest_size
            # count empty slots in chest
            for s in self.container.slots:
                if s != None and s.slot < chest_size:
                    empty -= 1
            return empty
        else:
            return 0

    def printContents(self, debug_lvl=1):
        if self.open():
            self.pybot.pdebug(f'Chest contents:', debug_lvl)
            empty = True
            for i in self.container.containerItems():
                empty = False
                self.pybot.pdebug(f'  {i.count:2} {i.displayName}', debug_lvl)
            if empty:
                self.pybot.pdebug(f'  (none)', debug_lvl)

    def printItems(self, items):
        self.pybot.pdebug(f'  Item List:', 1)
        for i in items:
            self.pybot.pdebug(f'    {i.slot:3}: {i.count:3} x {i.displayName}', 1)

    def itemCount(self, item_arg):

        item_type, item_name = self.pybot.itemTypeAndName(item_arg)

        count = 0
        inventory = self.container.containerItems()
        if inventory != []:
            # Count how many items we have of this type
            for i in inventory:
                if item_type == i.type:
                    count += i.count

        return count


    async def depositOneToChest(self, mineflayer_pathfinder, item_arg, count):

        # Go to chest
        await pathfind_to_goal(self.pybot.bot, self.object, "chest")

        # Open chest
        msg = self.open()
        if msg:
            return msg

        item_data = self.mcData.itemsByName[item_arg]
        item_name = item_data.displayName
        item_type = None

        invList = self.pybot.bot.inventory.items()
        for i in invList:
            if i.displayName == item_name:
                item_type = i.type

        # Check space in chest
        """if self.spaceAvailable() < 2:
            print('chest is full')
            return f'Chest is full.'"""

        # deposit item to chest
        try:
            self.container.deposit(item_type,None,count)
            self.close()
            return f'Successfully deposited all of item {item_name} in chest'
        except Exception as e:
            self.close()
            #print(f'failed to deposit in chest', e)
            return f"Inventory is not containing {item_name}."


    async def withdrawOneFromChest(self, mineflayer_pathfinder, item_arg, count):

        # Go to chest
        await pathfind_to_goal(self.pybot.bot, self.object, "chest")

        # Open chest
        msg = self.open()
        if msg:
            return msg

        item_data = self.mcData.itemsByName[item_arg]
        item_name = item_data.displayName
        item_type = None

        for i in self.container.containerItems():
            if i.displayName == item_name:
                item_type = i.type

        # Withdraw item from chest
        try:
            self.container.withdraw(item_type, None, count)
            self.close()
            return f'Successfully withdrawed all of item {item_name}'
        except Exception as e:
            self.close()
            #print(f'*** withdrawing 1 of item {item_type}', e)
            return f"Chest is not containing {item_name}."


    async def viewChest(self, mineflayer_pathfinder):

        # Go to chest
        await pathfind_to_goal(self.pybot.bot, self.object, "chest")

        # Open chest
        msg = self.open()
        if msg:
            return msg

        #print("start")

        items = self.container.containerItems()
        item_list = [f"{i.count}x {i.name}" for i in items] if items else []

        # Check space in chest
        space = self.spaceAvailable()
        space = 54
        if space < 2:
            space_message = "The chest is full."
        else:
            space_message = (str(space) + " slots are empty in chest." )

        self.close()

        #print(f"This items are in the chest: {item_list}. {space_message}")
        return f"This items are in the chest: {item_list}. {space_message}"


#==========================
# Furnace use
#==========================

async def smeltItem(bot, mcData, mineflayer_pathfinder, item, count):
    msg = ""
    item_data = mcData.itemsByName[item]
    smelt_item_name = item_data.displayName

    try:
        # find crafting table
        furnace_block_id = mcData.blocksByName['furnace'].id
        furnace_block = bot.findBlock({
            'matching': furnace_block_id,
            'maxDistance': 64,
            'count': 1
        })

        if not furnace_block:
            msg = (f"Wanted to smelt {item}, but no furnace found!")
            #print(msg)
            return msg

        # Move to goal
        await pathfind_to_goal(bot, furnace_block, "furnace")

        try:
            # Find item in inventory
            input_item = None
            item_count = None
            fuel_type = None
            fuel_count = None
            invList = bot.inventory.items()
            for i in invList:
                if i.displayName == smelt_item_name:
                    input_item = i.type
                    item_count = i.count
                elif i.displayName == "Coal":
                    fuel_type = i.type
                    fuel_count = i.count

            if input_item and item_count >= count:
                try:
                    furnace_window = bot.openFurnace(furnace_block)

                    try:
                        # Put item in input slot
                        furnace_window.putInput(input_item, None, count)

                        if fuel_type:
                            furnace_window.putFuel(fuel_type, None, fuel_count)
                            msg = (f"Started smelting {count}x {item}.")
                        else:
                            msg = (f"No fuel available for smelting")
                            #print(msg)
                            return msg

                        furnace_window.close()

                        #print(msg)
                        return msg
                    except Exception as e:
                        furnace_window.close()
                        msg = (f"The furnace is full and you have to clear it first.")
                        #print(msg)
                        return msg

                except Exception as e:
                    if 'furnace_window' in locals():
                        furnace_window.close()
                    msg = (f"Failed to open furnace.")
                    #print(msg)
                    return msg
            else:
                msg = (f"Don't have enough {item} to smelt")
                #print(msg)
                return msg
        except Exception as e:
            msg = (f"Don't have enough {item} to smelt")
            #print(msg)
            return msg
    except Exception as e:
        msg = (f"Wanted to smelt {item}, but no furnace found!")
        #print(msg)
        return msg


async def clearFurnace(bot, mcData, mineflayer_pathfinder, item, count):
    msg = ""
    collected_results = []

    try:
        # 1. Werkbank finden
        furnace_block_id = mcData.blocksByName['furnace'].id
        furnace_block = bot.findBlock({
            'matching': furnace_block_id,
            'maxDistance': 64,
            'count': 1
        })

        if not furnace_block:
            msg = (f"Wanted to smelt {item}, but no furnace found!")
            #print(msg)
            return msg

        # Move to goal
        await pathfind_to_goal(bot, furnace_block, "furnace")

        try:
            furnace_window = bot.openFurnace(furnace_block)

            for _ in range(20):
                if any(furnace_window.slots[i] for i in range(3)):
                    break
                await asyncio.sleep(0.1)

            # Check and clear slots separately
            # Slot 0: Input, Slot 1: Fuel, Slot 2: Output
            slot_labels = {0: "Input", 1: "Fuel", 2: "Output"}

            for slot_id, label in slot_labels.items():
                item_in_slot = furnace_window.slots[slot_id]

                if item_in_slot:
                    try:
                        bot.putAway(item_in_slot.slot)
                        collected_results.append(f"{item_in_slot.count}x {item_in_slot.name}")
                    except Exception as e:
                        print(f"Error while clearing furnace {label}: {e}")

            furnace_window.close()

            if collected_results:
                msg = (f"Successfully cleared furnace. "
                       f"You received {', '.join(collected_results)} from the furnace. .")
            else:
                msg = "Successfully cleared furnace, but it was empty."

            #print(msg)
            return msg

        except Exception as e:
            msg = (f"Failed to open/clear furnace.")
            #print(msg)
            return msg
    except Exception as e:
        msg = (f"Wanted to smelt {item}, but no furnace found!")
        #print(msg)
        return msg
