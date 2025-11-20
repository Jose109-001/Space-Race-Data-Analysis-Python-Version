import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("=" * 70)
print("🚀 SPACE RACE DATA ANALYSIS PROJECT 🚀")
print("Loading data from CSV file...")
print("=" * 70)

# ============================================
#  STEP 1: LOAD DATA FROM CSV FILE
# ============================================

print("\n📂 Loading mission_launches.csv...")

try:
    # Load the CSV file
    df = pd.read_csv("mission_launches.csv")
    print(f"Successfully loaded {len(df)} records")
    print(f"\nOriginal columns: {list(df.columns)}")
except FileNotFoundError:
    print("\n⚠️ ERROR: mission_launches.csv not found!")
    print("Please ensure the CSV file is in the same directory.")
    exit()
except Exception as e:
    print(f"\n⚠️ ERROR loading CSV: {e}")
    exit()

# ============================================
# STEP 2: PROCESS AND CLEAN THE DATA
# ============================================

print("\n" + "=" * 70)
print("PROCESSING DATA")
print("=" * 70)

# Rename columns to match our analysis
column_mapping = {
    "Organisation": "agency",
    "Location": "location_name",
    "Date": "date",
    "Detail": "rocket",
    "Rocket_Status": "rocket_status",
    "Price": "price",
    "Mission_Status": "status",
}

df = df.rename(columns=column_mapping)

# Extract country from location (last part after last comma)
df["country_full"] = df["location_name"].str.split(",").str[-1].str.strip()

# Create rocket_family from rocket detail (take first part before |)
df["rocket_family"] = df["rocket"].str.split("|").str[0].str.strip()

print(f"\nDataFrame processed with {len(df)} records")
print(f"Columns after mapping: {list(df.columns)}")

# ============================================
# STEP 3: DATA CLEANING AND PREPARATION
# ============================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove records with no date
df = df.dropna(subset=["date"])

# Extract temporal features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()
df["decade"] = (df["year"] // 10) * 10
df["day_of_week"] = df["date"].dt.day_name()

# Categorize success
success_keywords = ["Success", "Partial Failure"]
df["success"] = df["status"].isin(success_keywords)

# Map country codes to full names for common countries
country_mapping = {
    "USA": "United States",
    "China": "China",
    "Russia": "Russia",
    "Kazakhstan": "Kazakhstan",
    "French Guiana": "France",
    "India": "India",
    "Japan": "Japan",
    "New Zealand": "New Zealand",
}
df["country_full"] = df["country_full"].map(country_mapping).fillna(df["country_full"])

print(
    f"\nCleaned data: {len(df)} records from "
    f"{df['year'].min()} to {df['year'].max()}"
)

print("\nSample of processed data:")
sample_cols = ["rocket", "date", "year", "country_full", "agency", "status"]
print(df[sample_cols].head(10))

# ============================================
# STEP 4: WHO LAUNCHED THE MOST?
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS 1: MISSIONS BY COUNTRY AND ORGANIZATION")
print("=" * 70)

country_missions = df["country_full"].value_counts().head(15)
print("\nTop 15 Countries by Total Missions:")
print(country_missions)

org_missions = df["agency"].value_counts().head(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Countries
country_missions.plot(kind="barh", ax=ax1, color="steelblue")
ax1.set_title("Top 15 Countries by Space Missions", fontsize=14, fontweight="bold")
ax1.set_xlabel("Number of Missions", fontsize=11)
ax1.set_ylabel("Country", fontsize=11)

# Organizations
org_missions.plot(kind="barh", ax=ax2, color="coral")
ax2.set_title("Top 15 Organizations by Space Missions", fontsize=14, fontweight="bold")
ax2.set_xlabel("Number of Missions", fontsize=11)
ax2.set_ylabel("Organization", fontsize=11)

plt.tight_layout()
plt.show()

print("\nTop 15 Organizations:")
print(org_missions)

# ============================================
# STEP 5: LAUNCHES OVER TIME
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS 2: TEMPORAL TRENDS")
print("=" * 70)

# Missions per year
missions_per_year = df.groupby("year").size().reset_index(name="missions")

plt.figure(figsize=(16, 6))
plt.plot(
    missions_per_year["year"],
    missions_per_year["missions"],
    linewidth=2.5,
)
plt.fill_between(
    missions_per_year["year"],
    missions_per_year["missions"],
    alpha=0.3,
    color="lightblue",
)
plt.title("Space Missions Per Year (Historical Trend)", fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Missions", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Space Race: USA vs Russia vs China
major_powers = ["United States", "Russia", "China"]
df_powers = df[df["country_full"].isin(major_powers)]
power_launches = (
    df_powers.groupby(["year", "country_full"]).size().unstack(fill_value=0)
)

plt.figure(figsize=(16, 7))
for country in power_launches.columns:
    plt.plot(
        power_launches.index,
        power_launches[country],
        linewidth=2.5,
        marker="o",
        label=country,
        markersize=5,
    )

plt.title(
    "Space Race: USA vs Russia vs China Over Time", fontsize=16, fontweight="bold"
)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Launches", fontsize=12)
plt.legend(fontsize=12, loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Peak years
top_years = missions_per_year.nlargest(5, "missions")
print("\nTop 5 Years by Number of Launches:")
print(top_years.to_string(index=False))

# ============================================
# STEP 6: SEASONAL PATTERNS
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS 3: SEASONAL LAUNCH PATTERNS")
print("=" * 70)

# Launches by month
month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
month_counts = df["month_name"].value_counts().reindex(month_order)

print("\nLaunches by Month:")
print(month_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart
month_counts.plot(kind="bar", ax=ax1, color="teal", edgecolor="black")
ax1.set_title("Space Launches by Month", fontsize=14, fontweight="bold")
ax1.set_xlabel("Month", fontsize=11)
ax1.set_ylabel("Number of Launches", fontsize=11)
ax1.tick_params(axis="x", rotation=45)

# Pie chart
colors = plt.cm.Set3(range(12))
month_counts.plot(kind="pie", ax=ax2, autopct="%1.1f%%", colors=colors, startangle=90)
ax2.set_title("Launch Distribution by Month", fontsize=14, fontweight="bold")
ax2.set_ylabel("")

plt.tight_layout()
plt.show()

# Day of week analysis
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
day_counts = df["day_of_week"].value_counts().reindex(day_order)

print("\nLaunches by Day of Week:")
print(day_counts)

# ============================================
# STEP 7: SUCCESS RATE ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS 4: MISSION SUCCESS RATES")
print("=" * 70)

# Overall success rate
total_missions = len(df)
successful_missions = df["success"].sum()
success_rate = (successful_missions / total_missions) * 100

print(f"\nOverall Success Rate: {success_rate:.2f}%")
print(f"Successful Missions: {successful_missions:,}")
print(f"Failed/Partial Failure Missions: " f"{total_missions - successful_missions:,}")

# Success status breakdown
status_counts = df["status"].value_counts()
print("\nMission Status Distribution:")
print(status_counts)

# Visualize status distribution
plt.figure(figsize=(12, 6))
status_counts.head(8).plot(kind="bar", color="green", alpha=0.7, edgecolor="darkgreen")
plt.title("Mission Outcomes Distribution", fontsize=16, fontweight="bold")
plt.xlabel("Status", fontsize=12)
plt.ylabel("Number of Missions", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Success rate by decade
success_by_decade = df.groupby("decade")["success"].agg(["sum", "count"])
success_by_decade["success_rate"] = (
    success_by_decade["sum"] / success_by_decade["count"]
) * 100

print("\nSuccess Rate by Decade:")
print(success_by_decade)

plt.figure(figsize=(14, 6))
bars = plt.bar(
    success_by_decade.index,
    success_by_decade["success_rate"],
    color="green",
    alpha=0.7,
    edgecolor="darkgreen",
    linewidth=2,
)

# Color code bars
for i, bar in enumerate(bars):
    if success_by_decade["success_rate"].iloc[i] >= 90:
        bar.set_color("darkgreen")
    elif success_by_decade["success_rate"].iloc[i] >= 80:
        bar.set_color("green")
    else:
        bar.set_color("orange")

plt.title("Mission Success Rate by Decade", fontsize=16, fontweight="bold")
plt.xlabel("Decade", fontsize=12)
plt.ylabel("Success Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.axhline(
    y=success_rate,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Overall Average: {success_rate:.1f}%",
)
plt.legend(fontsize=11)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Success rate trend over time
success_by_year = df.groupby("year")["success"].agg(["sum", "count"])
success_by_year["rate"] = (success_by_year["sum"] / success_by_year["count"]) * 100
rolling_success = success_by_year["rate"].rolling(window=5, center=True).mean()

plt.figure(figsize=(16, 6))
plt.plot(
    success_by_year.index,
    success_by_year["rate"],
    alpha=0.3,
    linewidth=1,
    label="Yearly Success Rate",
    color="gray",
)
plt.plot(
    rolling_success.index,
    rolling_success,
    linewidth=3,
    color="darkgreen",
    label="5-Year Moving Average",
)
plt.title("Mission Success Rate Trend Over Time", fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Success Rate (%)", fontsize=12)
plt.ylim(0, 105)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# STEP 8: ROCKET FAMILIES AND TYPES
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS 5: ROCKET FAMILIES")
print("=" * 70)

# Most used rockets
top_rockets = df["rocket"].value_counts().head(15)
print("\nTop 15 Most Used Rockets:")
print(top_rockets)

plt.figure(figsize=(14, 7))
top_rockets.plot(kind="barh", alpha=0.7)
plt.title("Top 15 Most Frequently Launched Rockets", fontsize=16, fontweight="bold")
plt.xlabel("Number of Launches", fontsize=12)
plt.ylabel("Rocket Configuration", fontsize=12)
plt.tight_layout()
plt.show()

top_families = df["rocket_family"].value_counts().head(12)
print("\nTop 12 Rocket Families:")
print(top_families)

# ============================================
# STEP 9: ADVANCED VISUALIZATIONS
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS 6: ADVANCED VISUALIZATIONS")
print("=" * 70)

# Heatmap: Launches by Country and Decade
country_decade = pd.crosstab(df["country_full"], df["decade"])
top_countries_heatmap = country_decade.sum(axis=1).nlargest(15).index
heatmap_data = country_decade.loc[top_countries_heatmap]

plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_data, cmap="viridis", linewidths=0.5)
plt.title(
    "Space Missions Heatmap: Top Countries vs Decades", fontsize=16, fontweight="bold"
)
plt.xlabel("Decade", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.tight_layout()
plt.show()

# Stacked area chart: Top 5 countries over time
top_5_countries = df["country_full"].value_counts().head(5).index
df_top5 = df[df["country_full"].isin(top_5_countries)]
country_year_stack = (
    df_top5.groupby(["year", "country_full"]).size().unstack(fill_value=0)
)

plt.figure(figsize=(16, 8))
country_year_stack.plot.area(stacked=True, alpha=0.8, figsize=(16, 8))
plt.title(
    "Space Launch Trends: Top 5 Countries (Cumulative)", fontsize=16, fontweight="bold"
)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Launches", fontsize=12)
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# STEP 10: KEY INSIGHTS SUMMARY
# ============================================

print("\n" + "=" * 70)
print("KEY INSIGHTS SUMMARY")
print("=" * 70)

# Calculate trends
recent_decade = df[df["decade"] == df["decade"].max()]
old_decade = df[df["decade"] == df["decade"].min()]

recent_success = recent_decade["success"].mean() * 100 if len(recent_decade) > 0 else 0
old_success = old_decade["success"].mean() * 100 if len(old_decade) > 0 else 0

# Calculate additional metrics for summary
top_rocket_name = top_rockets.index[0] if len(top_rockets) > 0 else "N/A"
top_rocket_count = top_rockets.iloc[0] if len(top_rockets) > 0 else 0
trend_text = "IMPROVED" if recent_success > old_success else "DECLINED"
old_decade_label = (
    f"{old_decade['decade'].iloc[0]}s" if len(old_decade) > 0 else "early years"
)
recent_decade_label = (
    f"{recent_decade['decade'].iloc[0]}s" if len(recent_decade) > 0 else "recent years"
)
peak_year = missions_per_year.loc[missions_per_year["missions"].idxmax(), "year"]
recent_trend = (
    "increasing"
    if missions_per_year["missions"].iloc[-5:].mean()
    > missions_per_year["missions"].iloc[-10:-5].mean()
    else "stable"
)
spaceflight_status = (
    "emerging" if any("SpaceX" in str(x) for x in df["agency"].tail(200)) else "growing"
)

summary = f"""
📊 COMPREHENSIVE SPACE RACE ANALYSIS

1. DATASET OVERVIEW
   • Total Missions Analyzed: {len(df):,}
   • Time Period: {df['year'].min()} - {df['year'].max()}
   • Countries Represented: {df['country_full'].nunique()}
   • Organizations: {df['agency'].nunique()}

2. TOP PERFORMERS
   • Leading Country: {country_missions.index[0]} \
({country_missions.iloc[0]:,} missions)
   • Most Active Organization: {org_missions.index[0]} \
({org_missions.iloc[0]:,} missions)
   • Most Used Rocket: {top_rocket_name} ({top_rocket_count} launches)

3. MISSION SUCCESS
   • Overall Success Rate: {success_rate:.2f}%
   • Successful Missions: {successful_missions:,}
   • Trend: Success rates {trend_text}
     from {old_success:.1f}% in {old_decade_label}
     to {recent_success:.1f}% in {recent_decade_label}

4. LAUNCH PATTERNS
   • Peak Launch Year: {peak_year} \
({missions_per_year['missions'].max()} missions)
   • Most Popular Launch Month: {month_counts.idxmax()} \
({month_counts.max():,} launches)
   • Busiest Day of Week: {day_counts.idxmax()} \
({day_counts.max():,} launches)

5. HISTORICAL TRENDS
   • The Space Race era (1950s-1970s) saw rapid growth in launches
   • Success rates have generally improved with advancing technology
   • Recent years show {recent_trend} activity
   • Commercial spaceflight is {spaceflight_status}

 The journey from Sputnik to modern space exploration continues!
"""

print(summary)

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)

# ============================================
# 💾 STEP 11: EXPORT RESULTS
# ============================================

print("\n" + "=" * 70)
print("💾 EXPORTING ANALYSIS RESULTS")
print("=" * 70)

try:
    # Create exports directory if it doesn't exist
    import os

    if not os.path.exists("exports"):
        os.makedirs("exports")
        print("\n✅ Created 'exports' directory")

    # 1. Export processed data with all derived columns
    processed_file = "exports/processed_mission_data.csv"
    df.to_csv(processed_file, index=False)
    print(f"✅ Exported processed data: {processed_file}")

    # 2. Export country statistics
    country_stats = pd.DataFrame(
        {
            "Country": country_missions.index,
            "Total_Missions": country_missions.values,
            "Success_Rate_%": [
                (df[df["country_full"] == country]["success"].mean() * 100)
                for country in country_missions.index
            ],
        }
    )
    country_file = "exports/country_statistics.csv"
    country_stats.to_csv(country_file, index=False)
    print(f"✅ Exported country statistics: {country_file}")

    # 3. Export organization statistics
    org_stats = pd.DataFrame(
        {
            "Organization": org_missions.index,
            "Total_Missions": org_missions.values,
            "Success_Rate_%": [
                (df[df["agency"] == org]["success"].mean() * 100)
                for org in org_missions.index
            ],
        }
    )
    org_file = "exports/organization_statistics.csv"
    org_stats.to_csv(org_file, index=False)
    print(f"✅ Exported organization statistics: {org_file}")

    # 4. Export yearly trends
    yearly_stats = missions_per_year.copy()
    yearly_stats["Success_Rate_%"] = [
        (df[df["year"] == year]["success"].mean() * 100)
        for year in yearly_stats["year"]
    ]
    yearly_file = "exports/yearly_trends.csv"
    yearly_stats.to_csv(yearly_file, index=False)
    print(f"✅ Exported yearly trends: {yearly_file}")

    # 5. Export rocket statistics
    rocket_stats = pd.DataFrame(
        {
            "Rocket": top_rockets.index[:20],
            "Total_Launches": top_rockets.values[:20],
            "Success_Rate_%": [
                (df[df["rocket"] == rocket]["success"].mean() * 100)
                for rocket in top_rockets.index[:20]
            ],
        }
    )
    rocket_file = "exports/rocket_statistics.csv"
    rocket_stats.to_csv(rocket_file, index=False)
    print(f"✅ Exported rocket statistics: {rocket_file}")

    # 6. Export summary report as text file
    summary_file = "exports/analysis_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write(
            f"\n\nGenerated on: "
            f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
    print(f"✅ Exported summary report: {summary_file}")

    # 7. Export decade analysis
    decade_stats = success_by_decade.copy()
    decade_stats["total_missions"] = decade_stats["count"]
    decade_stats["successful_missions"] = decade_stats["sum"]
    decade_stats = decade_stats[
        ["total_missions", "successful_missions", "success_rate"]
    ]
    decade_stats.columns = ["Total_Missions", "Successful_Missions", "Success_Rate_%"]
    decade_file = "exports/decade_analysis.csv"
    decade_stats.to_csv(decade_file)
    print(f"✅ Exported decade analysis: {decade_file}")

    # 8. Export to Excel (all sheets in one file)
    try:
        excel_file = "exports/space_race_complete_analysis.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Processed Data", index=False)
            country_stats.to_excel(writer, sheet_name="Countries", index=False)
            org_stats.to_excel(writer, sheet_name="Organizations", index=False)
            yearly_stats.to_excel(writer, sheet_name="Yearly Trends", index=False)
            rocket_stats.to_excel(writer, sheet_name="Rockets", index=False)
            decade_stats.to_excel(writer, sheet_name="Decades")
        print(f"✅ Exported Excel workbook: {excel_file}")
    except ImportError:
        print("⚠️  Excel export skipped (install: pip install openpyxl)")

    print("\n" + "=" * 70)
    print("📁 All exports saved to 'exports/' directory")
    print("=" * 70)

except Exception as e:
    print(f"\n⚠️  Error during export: {e}")

print(
    """
    🌟 Thank you for exploring the Space Race data! 🌟
    
         🛸
        /   \\
       |  o  |
        \\_|_/
         | |
        /   \\
       
    📊 All visualizations have been generated.
    📈 Check the matplotlib windows for charts.
    💾 Analysis results exported to 'exports/' folder.
    🚀 Keep exploring the cosmos!
"""
)
