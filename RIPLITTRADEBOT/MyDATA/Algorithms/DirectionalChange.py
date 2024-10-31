def directional_change(prices, sigma):
    local_extremes = []
    upzig = True
    pending_extreme_index = 0
    pending_extreme_price = prices[0]

    for i in range(1, len(prices)):
        if upzig:
            if prices[i] > pending_extreme_price:
                pending_extreme_index = i
                pending_extreme_price = prices[i]
            elif prices[i] < pending_extreme_price * (1 - sigma):
                local_extremes.append((pending_extreme_index, pending_extreme_price))
                upzig = False
                pending_extreme_index = i
                pending_extreme_price = prices[i]
        else:
            if prices[i] < pending_extreme_price:
                pending_extreme_index = i
                pending_extreme_price = prices[i]
            elif prices[i] > pending_extreme_price * (1 + sigma):
                local_extremes.append((pending_extreme_index, pending_extreme_price))
                upzig = True
                pending_extreme_index = i
                pending_extreme_price = prices[i]

    return local_extremes

# Example usage
prices = [1, 3, 2, 4, 3, 5, 4, 6, 5, 7]
sigma = 0.1
extremes = directional_change(prices, sigma)
print("Local Extremes:", extremes)
