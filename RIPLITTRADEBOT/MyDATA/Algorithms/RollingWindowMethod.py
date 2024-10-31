def find_local_extremes(prices, order):
    local_tops = []
    local_bottoms = []

    for i in range(order, len(prices) - order):
        is_top = True
        is_bottom = True

        for j in range(1, order + 1):
            if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                is_top = False
            if prices[i] >= prices[i - j] or prices[i] >= prices[i + j]:
                is_bottom = False

        if is_top:
            local_tops.append(i)
        if is_bottom:
            local_bottoms.append(i)

    return local_tops, local_bottoms

# Example usage
prices = [1, 3, 7, 1, 2, 6, 3, 2, 1, 5]
order = 2
tops, bottoms = find_local_extremes(prices, order)
print("Local Tops:", tops)
print("Local Bottoms:", bottoms)
