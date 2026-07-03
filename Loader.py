import pandas as pd
import time


csv_path = "Material/Recipes.csv"


def get_recipe_from_csv(mcData, technical_name, bot):
    """
    Sucht ein Rezept in der CSV basierend auf dem technischen Namen (z.B. wooden_pickaxe).
    """
    try:
        # 1. Validierung: Ist mcData wirklich das richtige Objekt?
        if isinstance(mcData, str) or not hasattr(mcData, 'itemsByName'):
            return "Fehler: mcData wurde nicht korrekt übergeben."

        # 2. Umwandlung: technischer_name -> DisplayName
        # Minecraft-Daten nutzen (z.B. 'wooden_pickaxe' -> 'Wooden Pickaxe')
        display_name = None
        if technical_name in mcData.itemsByName:
            display_name = mcData.itemsByName[technical_name].displayName
        elif technical_name in mcData.blocksByName:
            display_name = mcData.blocksByName[technical_name].displayName

        if not display_name:
            # Fallback: Ersetze Unterstriche und mache Anfangsbuchstaben groß
            display_name = technical_name.replace('_', ' ').title()

        # 3. CSV laden
        df = pd.read_csv(csv_path)

        # 4. Suche in der CSV (OutputItem Spalte)
        # Wir nutzen .str.contains, falls in der CSV "Wood Pickaxe" statt "Wooden Pickaxe" steht
        search_term = display_name.replace("Wooden", "Wood")  # Kleiner Fix für deine CSV-Namen
        recipe_match = df[df['OutputItem'].str.lower() == search_term.lower()]

        # Falls exakter Match fehlschlägt, versuche Teilsuche
        if recipe_match.empty:
            recipe_match = df[df['OutputItem'].str.contains(search_term, case=False, na=False)]

        if recipe_match.empty:
            return f"Kein CSV-Eintrag für {display_name} gefunden."

        # 5. Find limiting resources
        items = bot.inventory.items()
        ingredients = []

        for _, row in recipe_match.iterrows():
            input_item = row['InputItem']
            required_qty = row['Qty']

            # Suche das Item im Inventar
            have_qty = 0
            for i in items:
                if input_item.lower() == i.name.lower():
                    have_qty = i.count
                    break

            # Berechne, wie viel fehlt
            missing = required_qty - have_qty

            if missing > 0:
                # Wir haben zu wenig
                ingredients.append(f"{missing}x {input_item}")

        if ingredients:
            return ", ".join(ingredients)

    except Exception as e:
        return f"Fehler in der Rezeptsuche: {str(e)}"


def write_log(log_file, final_history):
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- CONTEXT SENT TO AI ({time.ctime()}) ---\n")
            f.write(final_history)
            f.write("\n--------------------------\n")
    except Exception as e:
        print(f"Fehler beim Schreiben des AI-Logs: {e}")

