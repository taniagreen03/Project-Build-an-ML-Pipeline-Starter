import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################

def test_row_count(data):
    """
    Test to confirm the number of rows in a dataset
    is greater than 15,000 and less than 1,000,000
    """
    assert 15000 < data.shape[0] < 1000000
def test_price_range(data, min_price, max_price):
    """
    Test to check if every value (price)
    is between the minimum and maxixmum value (price)
     """
    assert data['price'].between(min_price, max_price).all()


# def test_price_range(data, min_price, max_price):
#     """
#     Not my original code. helper code to debug Assertion error
#     Test to check if every price value is between min_price and max_price.
#     """
#     out_of_range = data[~data["price"].between(float(min_price), float(max_price))]
#
#     print("\n✅ Checking 'price' column:")
#     print(f"✅ min_price: {min_price} ({type(min_price)})")
#     print(f"✅ max_price: {max_price} ({type(max_price)})")
#     print(f"✅ Data type of 'price' column: {data['price'].dtype}")
#
#     if not out_of_range.empty:
#         print(f"❌ Some prices are out of range:\n{out_of_range[['id', 'price']]}")  # Print only price column
#
#     assert out_of_range.empty, f"❌ Test failed: Some prices are out of range:\n{out_of_range[['id', 'price']]}"
#

