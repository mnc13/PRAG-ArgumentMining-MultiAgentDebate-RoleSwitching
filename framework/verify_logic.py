def simulate_slicing(offset, limit, total_claims):
    all_claims = list(range(total_claims))
    start_idx = offset
    end_idx = offset + limit if limit is not None else len(all_claims)
    sliced_claims = all_claims[start_idx:end_idx]
    return sliced_claims

def test_cases():
    # Case 1: Offset 30, Limit 10 (Claims 30-39)
    res1 = simulate_slicing(30, 10, 125)
    print(f"Test 30/10: Range {res1[0]}-{res1[-1]}, Size {len(res1)}")
    assert res1[0] == 30 and res1[-1] == 39 and len(res1) == 10
    
    # Case 2: Offset 90, Limit 10 (Claims 90-99)
    res2 = simulate_slicing(90, 10, 125)
    print(f"Test 90/10: Range {res2[0]}-{res2[-1]}, Size {len(res2)}")
    assert res2[0] == 90 and res2[-1] == 99 and len(res2) == 10
    
    # Case 3: Resume scenario within window 90-100
    # Simulate the loop behavior with processed check
    processed_ids = set(range(90, 96)) # 90-95 are done
    window = simulate_slicing(90, 10, 125)
    remaining_to_run = [c for c in window if c not in processed_ids]
    print(f"Resume 90/10 (90-95 done): Will run {remaining_to_run}")
    assert remaining_to_run == [96, 97, 98, 99]
    
    print("\nAll logical tests passed! The slicing logic correctly bounds the window.")

if __name__ == "__main__":
    test_cases()
