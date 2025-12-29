import requests
import aiohttp
import asyncio


# ---------- SYNC ----------

def get_current_temperature_sync(city: str, api_key: str) -> float:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        raise ValueError(data)

    return data["main"]["temp"]


# ---------- ASYNC ----------

async def get_current_temperature_async(city: str, api_key: str) -> float:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if response.status != 200:
                raise ValueError(data)
            return data["main"]["temp"]
