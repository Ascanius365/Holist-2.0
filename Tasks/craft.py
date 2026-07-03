from javascript import require
from Loader import get_recipe_from_csv
from Tasks.movement import pathfind_to_goal

pathfinder_mod = require('mineflayer-pathfinder')
Movements = pathfinder_mod.Movements
GoalNear = pathfinder_mod.goals.GoalNear


async def craft(bot, mcData, mineflayer_pathfinder, item_arg, count):

    crafting_table_id = mcData.blocksByName['crafting_table'].id
    crafting_table = bot.findBlock({
        'matching': crafting_table_id,
        'maxDistance': 64,
        'count': 1
    })

    if not crafting_table:
        msg = (f"Wanted to craft {item_arg}, but no working bench found!")
        #print(msg)
        return msg

    await pathfind_to_goal(bot, crafting_table, item_arg)

    bot.lookAt(crafting_table.position.offset(0.5, 0.5, 0.5))

    try:
        item_data = mcData.itemsByName[item_arg]
        item_name = item_data.displayName
    except Exception as e:
        msg = (f"Item '{item_arg}' is unknown and cant be crafted! Try another name.")
        #print(msg)
        return msg

    item_id = item_data.id
    recipes = bot.recipesFor(item_id, None, count, crafting_table)

    if not recipes:
        msg = (f"This item '{item_name}' is not a recipe!")
        #print(msg)
        return msg

    recipe = recipes[0]

    #print(f"🔨 Start Crafting: {count}x {item_name}...")

    item_count = 0
    msg = ""
    while item_count < count:
        try:
            bot.craft(recipe, 1, crafting_table)

            invList = bot.inventory.items()
            for i in invList:
                if i.displayName == item_name:
                    item_count = i.count

            msg = (f"Successfully crafted {item_count}x {item_name}!")

        except Exception as e:
            #print(f"❌ Fehler während bot.craft: {e}")

            # Limiting Materials ---
            limit = get_recipe_from_csv(mcData, item_name, bot)

            if item_count == 0:
                msg += (f"You can't craft {item_name}, because you don't have the limiting materials: {limit}!")
            else:
                msg += (f"To craft another {item_name} you need the limiting materials: {limit}!")
            break

    #print(msg)
    return msg
