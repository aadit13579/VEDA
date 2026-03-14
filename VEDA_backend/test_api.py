def test_handshake():
    mock_layout_output = {
        "layout_data": [{
            "regions": [
                {"bbox": [10, 10, 50, 50], "type": "text"},
                {"bbox": [60, 10, 100, 50], "type": "table"}
            ]
        }]
    }
    
    try:
        result = process_spatial_sort(mock_layout_output)
        assert "reading_order" in result["layout_data"][0]["regions"][0]
        print("Structure check passed!")
    except KeyError as e:
        print(f"Structure check failed! Missing key: {e}")