# Weather Insights Analysis ☂️✨

Welcome to the **Weather Insights Analysis** project! This project provides a comparative visualization of weather data using pairplots to explore the relationships between variables such as temperature, humidity, wind speed, cloud cover, and pressure under two different scenarios:

1. **Rows Dropped** (Missing values removed)
2. **Missing Values Filled** (Replaced with the column mean)

---

## Pairplot Visualizations 🎨

### 1. Rows Dropped ❌⛈
The first pairplot visualizes the dataset after rows containing missing values were removed.

![Rows Dropped Pairplot](./Screenshot%20(25).png)

**Key Observations**:
- Temperature distribution shows a clear separation for rainy 🌧️ vs. non-rainy ☀️ weather.
- Humidity has a strong relationship with rain presence.

### 2. Missing Values Filled ✔️☀️
The second pairplot visualizes the dataset after missing values were filled with the column mean.

![Missing Values Filled Pairplot](./Screenshot%20(26).png)

**Key Observations**:
- Patterns are more uniform compared to the dropped rows scenario.
- Replacing missing values ensures that no data is lost, helping maintain data integrity for analysis.

---

## Features of the Project 📊

1. **Data Cleaning**: Missing values are handled in two ways: by dropping rows or filling them with the column mean.
2. **Visualization**: Interactive and visually appealing pairplots for comparative analysis.
3. **Insights**: Easily identify relationships between weather variables and rainfall occurrence.

---

## How to Use ⚙️

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/weather-insights.git
   ```
2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the visualization script:
   ```bash
   python visualize_weather_data.py
   ```

---

## Libraries Used 📊

- **Pandas**: For data manipulation.
- **Seaborn**: For creating pairplots.
- **Matplotlib**: For custom visualization enhancements.

---

## Future Improvements 🎯

- Add interactive visualizations using Plotly.
- Incorporate more advanced imputation techniques for missing data.
- Automate the generation of insights with AI.

---

Thank you for exploring this project! 🌟 Feel free to contribute and share your thoughts. Let’s analyze weather data better, together! ☁️☀️

