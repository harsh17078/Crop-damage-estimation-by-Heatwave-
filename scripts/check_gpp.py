
import ee

ee.Initialize(project="crop-heatwave-minor-project")

collection = ee.ImageCollection("MODIS/061/MOD17A2H")

print("Checking historical GPP coverage...\n")

for year in range(2001, 2006):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    yearly_collection = collection.filterDate(start, end)

    try:
        count = yearly_collection.size().getInfo()
        print(f"{year}: {count} images")

        if count > 0:
            print(
                "  Bands:",
                yearly_collection.first().bandNames().getInfo()
            )

    except Exception as e:
        print(f"{year}: ERROR - {e}")

