from Tasks.inventory import wieldItem

async def eat(bot, mcData, item):
    if bot.food > 18:
        msg = "Wanted to eat, but hunger is satisfied already (Food Level > 18). Don't eat more!"
        #print(msg)
        return True, msg

    # Wield food in hand
    msg = wieldItem(bot, mcData, item)

    if msg:
        msg = f"Wanted to eat {item}, but not in inventory!"
        #print(msg)
        return msg


    #print(f'Ate {item}...')
    await bot.consume()
    return f"Successfully ate {item}."
