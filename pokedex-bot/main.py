import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
from pokedex import entrenar_modelo, predecir_combate, generar_dataset, pd, obtener_pokemon

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

if not TOKEN:
    raise ValueError("No se encontró el token de Discord en .env")

# Configuración de intents
intents = discord.Intents.default()
intents.message_content = True  # Necesario para leer mensajes
bot = commands.Bot(command_prefix='!', intents=intents)

# Variable global para el modelo
model = None

@bot.event
async def on_ready():
    global model
    print(f'Bot conectado como {bot.user.name}')
    
    # Cargar o generar dataset
    if os.path.exists("pokemon_dataset.csv"):
        print("Cargando dataset...")
        df = pd.read_csv("pokemon_dataset.csv")
    else:
        print("Generando dataset... (esto puede tardar)")
        df = await asyncio.to_thread(generar_dataset, 500)
        df.to_csv("pokemon_dataset.csv", index=False)
    
    model = entrenar_modelo(df)
    print("✅ Modelo listo!")

@bot.command(name='predict')
async def predict(ctx, *, argumentos: str):
    """Predice el resultado de un combate Pokémon (uso: !predict pokemon1, pokemon2)"""
    if model is None:
        await ctx.send("⚠️ El modelo aún se está cargando. Intenta en unos segundos.")
        return
    
    try:
        # Procesamiento mejorado del input
        argumentos = argumentos.replace(" ", "")  # Elimina espacios
        pokemons = [p.strip() for p in argumentos.split(",")]
        
        if len(pokemons) != 2:
            await ctx.send("**❌ Formato incorrecto**\nUsa: `!predict pokemon1,pokemon2`")
            return
            
        pokemon1, pokemon2 = pokemons
        
        # Validación básica de nombres
        if not all(p.isalpha() for p in pokemons):
            await ctx.send("**⚠️ Nombres inválidos**\nSolo usa letras (ej: `pikachu,charizard`)")
            return
        
        # Obtener datos de los Pokémon
        p1 = await asyncio.to_thread(obtener_pokemon, pokemon1)
        p2 = await asyncio.to_thread(obtener_pokemon, pokemon2)
        
        if not p1 or not p2:
            await ctx.send("**❌ Uno o ambos Pokémon no fueron encontrados**")
            return
        
        # Obtener predicción
        resultado = await asyncio.to_thread(
            predecir_combate,
            model,
            f"{pokemon1},{pokemon2}"
        )
        
        # Crear embed con los sprites
        embed = discord.Embed(
            title=f"🔮 {p1['name']} vs {p2['name']}",
            description=f"```diff\n{resultado}\n```",
            color=discord.Color.blue()
        )
        
        embed.set_thumbnail(url=p1['sprite'])
        embed.set_image(url=p2['sprite'])
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"**❌ Error inesperado:**\n```{str(e)}```")

# Añade este nuevo comando junto a los existentes
@bot.command(name='info')
async def pokemon_info(ctx, *, nombre_pokemon: str):
    """Muestra información detallada de un Pokémon (uso: !info pikachu)"""
    try:
        # Obtener datos del Pokémon
        pokemon = await asyncio.to_thread(obtener_pokemon, nombre_pokemon)
        
        if not pokemon:
            await ctx.send(f"**❌ Pokémon '{nombre_pokemon}' no encontrado**")
            return
        
        # Crear embed con la información
        embed = discord.Embed(
            title=f"#{pokemon.get('id', '?')} {pokemon['name']}",
            color=discord.Color.gold()
        )
        
        # Configurar imagen principal (sprite frontal)
        embed.set_image(url=pokemon['sprite'])
        
        # Añadir campos con la información
        embed.add_field(name="🔹 Tipos", value=", ".join([t.capitalize() for t in pokemon['types']]), inline=True)
        
        # Estadísticas
        stats = pokemon['stats']
        embed.add_field(
            name="⚔️ Estadísticas", 
            value=(
                f"**HP:** {stats['hp']}\n"
                f"**Ataque:** {stats['attack']}\n"
                f"**Defensa:** {stats['defense']}\n"
                f"**Velocidad:** {stats['speed']}"
            ), 
            inline=True
        )
        
        # Movimientos (mostramos solo los primeros 5)
        moves = ", ".join([m.capitalize() for m in pokemon['moves'][:5]])
        if len(pokemon['moves']) > 5:
            moves += f" (+{len(pokemon['moves']) - 5} más)"
            
        embed.add_field(name="✨ Movimientos", value=moves, inline=False)
        
        # Footer con datos adicionales
        embed.set_footer(text=f"Información de Pokémon solicitada por {ctx.author.display_name}")
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"**❌ Error al obtener información:**\n```{str(e)}```")

@bot.command(name='compare')
async def compare_stats(ctx, stat: str, pokemon1: str, pokemon2: str):
    """Compara una estadística entre dos Pokémon (uso: !compare stat pokemon1 pokemon2)"""
    try:
        # Validar el stat solicitado
        valid_stats = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
        stat = stat.lower()
        
        if stat not in valid_stats:
            await ctx.send(
                f"**❌ Estadística inválida**\n"
                f"Estadísticas válidas: {', '.join(valid_stats)}\n"
                f"Ejemplo: `!compare attack pikachu charizard`"
            )
            return

        # Obtener datos de los Pokémon
        p1 = await asyncio.to_thread(obtener_pokemon, pokemon1)
        p2 = await asyncio.to_thread(obtener_pokemon, pokemon2)
        
        if not p1 or not p2:
            await ctx.send("**❌ Uno o ambos Pokémon no fueron encontrados**")
            return

        # Obtener valores de la estadística
        stat_p1 = p1['stats'].get(stat, 0)
        stat_p2 = p2['stats'].get(stat, 0)
        
        # Determinar el ganador
        if stat_p1 > stat_p2:
            result = f"**{p1['name']}** es mejor en {stat} ({stat_p1} > {stat_p2})"
        elif stat_p2 > stat_p1:
            result = f"**{p2['name']}** es mejor en {stat} ({stat_p2} > {stat_p1})"
        else:
            result = f"**Empate** en {stat} ({stat_p1} = {stat_p2})"

        # Crear embed con la comparación
        embed = discord.Embed(
            title=f"⚔️ Comparando {stat}",
            description=result,
            color=discord.Color.blue()
        )
        
        # Añadir campos con las stats de cada Pokémon
        embed.add_field(
            name=f"{p1['name']}",
            value=f"**{stat.capitalize()}:** {stat_p1}",
            inline=True
        )
        
        embed.add_field(
            name=f"{p2['name']}",
            value=f"**{stat.capitalize()}:** {stat_p2}",
            inline=True
        )
        
        # Añadir sprites pequeños como thumbnails
        embed.set_thumbnail(url=p1['sprite'])
        embed.set_image(url=p2['sprite'])
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"**❌ Error en la comparación:**\n```{str(e)}```")

bot.run(TOKEN)