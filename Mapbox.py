import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="3D Map Wow")

st.title("u/ 3D Hexagon Layer Visualizer")
st.markdown("Biểu đồ mật độ dân số/dữ liệu giả lập dưới dạng 3D interactive.")

# 1. Tạo dữ liệu giả lập (Tọa độ quanh TP.HCM)
# Lat: 10.7 ~ 10.8, Lon: 106.6 ~ 106.7
DATA_URL = {
    "lat": np.random.normal(10.776, 0.05, 1000),
    "lon": np.random.normal(106.700, 0.05, 1000)
}
df = pd.DataFrame(DATA_URL)

# 2. Cấu hình PyDeck Layer
layer = pdk.Layer(
    "HexagonLayer",
    df,
    get_position=["lon", "lat"],
    auto_highlight=True,
    elevation_scale=100,  # Độ cao của cột
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,                 # Dựng khối 3D
    coverage=1,
    radius=200,                    # Bán kính mỗi ô lục giác
    get_fill_color="[255, (1 - elevationValue / 500) * 255, 0, 180]", # Màu gradient theo độ cao
)

# 3. Cấu hình View (Góc nhìn camera)
view_state = pdk.ViewState(
    longitude=106.700,
    latitude=10.776,
    zoom=11,
    min_zoom=5,
    max_zoom=15,
    pitch=60.5,  # Góc nghiêng để nhìn thấy 3D
    bearing=-27.36,
)

# 4. Render
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Mật độ: {elevationValue}"},
    map_style="mapbox://styles/mapbox/dark-v10", # Chế độ tối cho ngầu (Cần mapbox token nếu muốn full đẹp, mặc định vẫn chạy đc basic)
)

st.pydeck_chart(r)