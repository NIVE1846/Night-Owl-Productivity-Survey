# 🦉 Night Owl Productivity Survey App

An interactive web-based survey application to study how individuals' preferred active hours relate to productivity, distraction levels, and sleep patterns.

## 🎯 Objective

To analyze whether night-time productivity is supported by focus and efficiency or offset by increased distraction and fatigue.

## ✨ Features

- **Interactive Survey Form** with validation
- **Animated Gradient Background** (night-to-day theme)
- **Real-time Data Storage** (CSV format)
- **Comprehensive Analysis Dashboard** with:
  - Descriptive statistics
  - Chronotype comparisons (Night Owl vs Early Bird)
  - Correlation analysis
  - Interactive visualizations
  - Data export functionality

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Survey Sections

1. **Demographics**: Name, email, age, occupation
2. **Work & Activity Timing**: Peak productivity hours, chronotype
3. **Sleep Patterns**: Bedtime, wake time, duration, quality
4. **Digital Habits**: Device usage, social media, distraction levels
5. **Productivity & Focus**: Self-rated productivity, focus duration, stress, energy patterns

## 📈 Analysis Features

- **Overview Tab**: Key metrics, distributions, demographics
- **Chronotype Analysis**: Night Owl vs Early Bird comparisons
- **Correlations**: Heatmaps and scatter plots with trendlines
- **Raw Data**: View and download complete dataset

## 📁 Project Structure

```
PythonStreamlit/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── data/
│   └── survey_responses.csv  # Survey data (auto-generated)
└── assets/               # Images/resources (optional)
```

## 🔍 Key Insights Analyzed

- Productivity differences between chronotypes
- Impact of sleep duration on productivity
- Correlation between device usage and distraction
- Relationship between social media and focus time
- Age and occupation patterns

## 🛠️ Technologies

- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **Python 3.8+**: Core language

## 📝 Data Privacy

All survey responses are stored locally in CSV format. No data is transmitted to external servers.

---

**Built with ❤️ using Python & Streamlit**
