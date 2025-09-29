import pandas as pd
import pytest
from intraday_clustering_enhanced import get_weekly_pattern_analysis

def test_weekly_pattern_chronological_order():
    """
    Tests that the get_weekly_pattern_analysis function correctly sorts
    weekdays in chronological order, not alphabetical.
    """
    # 1. Create a sample DataFrame with non-chronological dates
    data = {
        'date': pd.to_datetime([
            '2023-01-06',  # Friday
            '2023-01-02',  # Monday
            '2023-01-04',  # Wednesday
        ]),
        'cluster': [0, 1, 0]
    }
    sample_df = pd.DataFrame(data)

    # 2. Run the function to get the weekly pattern analysis
    weekday_pct = get_weekly_pattern_analysis(sample_df)

    # 3. Get the resulting order of weekdays from the DataFrame index
    actual_order = weekday_pct.index.tolist()

    # 4. Define the expected chronological order
    expected_order = ['Monday', 'Wednesday', 'Friday']

    # 5. Assert that the actual order matches the expected chronological order
    print(f"Actual order: {actual_order}")
    print(f"Expected order: {expected_order}")

    assert actual_order == expected_order, \
        f"Weekdays are not in chronological order. Got {actual_order}, expected {expected_order}."

if __name__ == "__main__":
    pytest.main([__file__])