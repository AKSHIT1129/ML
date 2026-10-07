import base64

def b64(fn):
    with open(fn, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

day = b64('earth_atmos_2048.jpg')
spec = b64('earth_specular_2048.jpg')
clouds = b64('earth_clouds_1024.png')
lights = b64('earth_lights_2048.png')

with open('earth_textures.js', 'w', encoding='utf-8') as f:
    f.write(f'window.EARTH_TEXTURE_DAY = "data:image/jpeg;base64,{day}";\n')
    f.write(f'window.EARTH_TEXTURE_SPECULAR = "data:image/jpeg;base64,{spec}";\n')
    f.write(f'window.EARTH_TEXTURE_CLOUDS = "data:image/png;base64,{clouds}";\n')
    f.write(f'window.EARTH_TEXTURE_LIGHTS = "data:image/png;base64,{lights}";\n')

print('Generated earth_textures.js successfully')
