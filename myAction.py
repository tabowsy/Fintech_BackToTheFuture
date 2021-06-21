import numpy as np

# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell before buy to get cash each day
    # default
    cash = 1000
    hold = 0
    # user definition
    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    stockHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    
    for day in range( 0, dataLen-nextDay ) :
        dayPrices = priceMat[day]  # Today price of each stock
        nextDayPrices = priceMat[ day + nextDay ]  # Next day price of each stock
        
        if day > 0:
            stockHolding[day] = stockHolding[day-1]  # The stock holding from the previous action day
        
        buyStock = -1  # which stock should buy. No action when is -1
        buyPrice = 0  # use how much cash to buy
        sellStock = []  # which stock should sell. No action when is null
        sellPrice = []  # get how much cash from sell
        bestPriceDiff = 0  # difference in today price & next day price of "buy" stock
        stockCurrentPrice = 0  # The current price of "buy" stock
        
        # Check next day price to "sell"
        for stock in range(stockCount) :
            todayPrice = dayPrices[stock]  # Today price
            nextDayPrice = nextDayPrices[stock]  # Next day price
            holding = stockHolding[day][stock]  # how much stock you are holding
            
            if holding > 0 :  # "sell" only when you have stock holding
                if nextDayPrice < todayPrice*(1+transFeeRate) :  # next day price < today price, should "sell"
                    sellStock.append(stock)
                    # "Sell"
                    sellPrice.append(holding * todayPrice)
                    cash = holding * todayPrice*(1-transFeeRate) # Sell stock to have cash
                    stockHolding[day][sellStock] = 0
        
        # Check next day price to "buy"
        if cash > 0 :  # "buy" only when you have cash
            for stock in range(stockCount) :
                todayPrice = dayPrices[stock]  # Today price
                nextDayPrice = nextDayPrices[stock]  # Next day price
                
                if nextDayPrice > todayPrice*(1+transFeeRate) :  # next day price > today price, should "buy"
                    diff = nextDayPrice - todayPrice*(1+transFeeRate)
                    if diff > bestPriceDiff :  # this stock is better
                        bestPriceDiff = diff
                        buyStock = stock
                        stockCurrentPrice = todayPrice
            # "Buy" the best stock
            if buyStock >= 0 :
                buyPrice = cash
                stockHolding[day][buyStock] = cash*(1-transFeeRate) / stockCurrentPrice # Buy stock using cash
                cash = 0
                
        # Save your action this day
        if buyStock >= 0 or len(sellStock) > 0 :
            action = []
            if len(sellStock) > 0 :
                for i in range( len(sellStock) ) :
                    action = [day, sellStock[i], -1, sellPrice[i]]
                    actionMat.append( action )
            if buyStock >= 0 :
                action = [day, -1, buyStock, buyPrice]
                actionMat.append( action )

    return actionMat

# A DP-based approach to obtain the optimal return=
def myAction01(priceMat, transFeeRate):
    days, types = priceMat.shape
    cash = 1000
    use_cash = 4
    adjust1 = (1 - transFeeRate)
    adjust2 = 100.0 / 99.0
    # dp_record = np.zeros((days, 5), dtype=float)
    dp_record = [[0 for j in range(5)] for i in range(days)]
    trans_record = [[0 for j in range(6)] for i in range(days)]
    action_matrix = []

    # init
    dp_record[0] = [cash / priceMat[0][0] * adjust1, cash / priceMat[0][1] * adjust1, cash / priceMat[0][2] * adjust1,
                    cash / priceMat[0][3] * adjust1, cash]
    trans_record[0] = [0, 1, 2, 3, 1000.0, 4]
    cash_have = cash

    for i in range(1, days):
        sell_idx = -1  # no stock sold
        for j in range(4):
            cash_new = dp_record[i - 1][j] * priceMat[i][j] * adjust1  # yesterday stock equal how much cash today
            if cash_new > cash_have:
                cash_have = cash_new
                sell_idx = j

        if cash_have > dp_record[i - 1][-1]:
            dp_record[i][-1] = cash_have
            trans_record[i][-1] = sell_idx
            trans_record[i][
                -2] = cash_have * adjust2  # turn back to the cash that can be determine later in the other process
        else:
            dp_record[i][-1] = dp_record[i - 1][-1]
            trans_record[i][-1] = use_cash
            trans_record[i][-2] = trans_record[i - 1][-2]

        for j in range(4):
            stock_have = cash_have / priceMat[i][j] * adjust1
            if stock_have > dp_record[i - 1][j]:
                dp_record[i][j] = stock_have
                trans_record[i][j] = sell_idx  # change from which stock or cash
            else:
                dp_record[i][j] = dp_record[i - 1][j]
                trans_record[i][j] = j



    sell_from = trans_record[-1][-1]
    buy_to = -1
    for i in range(days - 1, -1, -1):
        # print(i, trans_record[i], dp_record[i], end='\n')
        if i == 0:
            sell_from = -1
        if not (sell_from ^ use_cash): #let 4 to -1
            sell_from = -1
        if not (buy_to ^ use_cash):
            buy_to = -1

        if (sell_from ^ buy_to):  # if transfer then add to actionMRX
            action_matrix.append([i, sell_from, buy_to, trans_record[i][-2]])
            # print(i,[i, sell_from, buy_to, trans_record[i][-2]])

        buy_to = sell_from
        sell_from = trans_record[i - 1][buy_to]

    actionMat =  action_matrix[::-1]
    return actionMat



# An approach that allow non-consecutive K days to hold all cash without any stocks
def myAction02(priceMat, transFeeRate, K):
    days, types = priceMat.shape
    cash = 1000
    use_cash = 4
    adjust1 = (1 - transFeeRate)
    adjust2 = 100.0 / 99.0
    # dp_record = np.zeros((days, 5), dtype=float)
    dp_record = [[0 for j in range(5)] for i in range(days)]
    trans_record = [[0 for j in range(6)] for i in range(days)]
    action_matrix = []

    # init
    dp_record[0] = [cash / priceMat[0][0] * adjust1, cash / priceMat[0][1] * adjust1, cash / priceMat[0][2] * adjust1,
                    cash / priceMat[0][3] * adjust1, cash]
    trans_record[0] = [0, 1, 2, 3, 1000.0, 4]
    cash_have = cash

    for i in range(1, days):
        sell_idx = -1  # no stock sold
        for j in range(4):
            cash_new = dp_record[i - 1][j] * priceMat[i][j] * adjust1  # yesterday stock equal how much cash today
            if cash_new > cash_have:
                cash_have = cash_new
                sell_idx = j

        if cash_have > dp_record[i - 1][-1]:
            dp_record[i][-1] = cash_have
            trans_record[i][-1] = sell_idx
            trans_record[i][
                -2] = cash_have * adjust2  # turn back to the cash that can be determine later in the other process
        else:
            dp_record[i][-1] = dp_record[i - 1][-1]
            trans_record[i][-1] = use_cash
            trans_record[i][-2] = trans_record[i - 1][-2]

        for j in range(4):
            stock_have = cash_have / priceMat[i][j] * adjust1
            if stock_have > dp_record[i - 1][j]:
                dp_record[i][j] = stock_have
                trans_record[i][j] = sell_idx  # change from which stock or cash
            else:
                dp_record[i][j] = dp_record[i - 1][j]
                trans_record[i][j] = j

    sell_from = trans_record[-1][-1]
    buy_to = -1
    for i in range(days - 1, -1, -1):
        # print(i, trans_record[i], dp_record[i], end='\n')
        if i == 0:
            sell_from = -1
        if not (sell_from ^ use_cash):  # let 4 to -1
            sell_from = -1
        if not (buy_to ^ use_cash):
            buy_to = -1

        if (sell_from ^ buy_to):  # if transfer then add to actionMRX
            action_matrix.append([i, sell_from, buy_to, trans_record[i][-2]])
            # print(i,[i, sell_from, buy_to, trans_record[i][-2]])

        buy_to = sell_from
        sell_from = trans_record[i - 1][buy_to]

    # actionMat = action_matrix
    actionMat = action_matrix[::-1]

    #TODO: deal with cash holding
    trans_days = days - K
    # print(trans_days)

    r_list = [] #[[buy day, sell day, return rate],[]...]
    r = []
    last_cash = 0
    for data in actionMat:
        if data[1] == -1:
            # print('buy stock',data)
            last_day = data[0]
            last_cash = data[3]
            r.append(data)


        if data[2] == -1:
            # print('sell stock',data)
            r.append(data)
            r.append(data[3]/last_cash)
            r.append(data[0]-last_day)
            r_list.append(r)
            #init
            r = []
    # pick the best nth r and satisfy cash holding day
    r_list = sorted(r_list, key=lambda x:x[2], reverse=True)
    count = 0
    rr_list = []
    for r in r_list:
        count += r[3]
        if count < trans_days:
            rr_list.append(r[0])
        else:
            count -= r[3]
            # print(count)
            break
    rr_list = sorted(rr_list, key=lambda x:x[0])

    ans = []
    find = 0
    for i,a in enumerate(actionMat):
        # if i == 0:
        #     ans.remove(a)

        if a in rr_list:
            ans.append(a)
            find = 1
        else:
            if a[2]!=-1 and find == 1 :
                ans.append(a)
            elif a[2] == -1 and find == 1:
                ans.append(a)
                find = 0
            else:
                pass
    # for i in ans:
    #     print(i)
    return ans

# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
    total_days, types = priceMat.shape
    zero = np.ones((K,types))
    for i in range(K):
        zero[i] *= 1/(i+1)

    last_record = 0
    for mask in range(total_days-K+1):
        priceMat_masked = np.concatenate((priceMat[0:mask],zero,priceMat[mask+K:]), axis=0)
        days, types = priceMat_masked.shape
        cash = 1000
        use_cash = 4
        adjust1 = (1 - transFeeRate)
        adjust2 = 100.0 / 99.0
        # dp_record = np.zeros((days, 5), dtype=float)
        dp_record = [[0 for j in range(5)] for i in range(days)]
        trans_record = [[0 for j in range(6)] for i in range(days)]


        # init
        dp_record[0] = [cash / priceMat_masked[0][0] * adjust1, cash / priceMat_masked[0][1] * adjust1, cash / priceMat_masked[0][2] * adjust1,
                        cash / priceMat_masked[0][3] * adjust1, cash]
        trans_record[0] = [0, 1, 2, 3, 1000.0, 4]
        cash_have = cash

        for i in range(1, days):
            sell_idx = -1  # no stock sold
            for j in range(4):
                cash_new = dp_record[i - 1][j] * priceMat_masked[i][j] * adjust1  # yesterday stock equal how much cash today
                if cash_new > cash_have:
                    cash_have = cash_new
                    sell_idx = j

            if cash_have > dp_record[i - 1][-1]:
                dp_record[i][-1] = cash_have
                trans_record[i][-1] = sell_idx
                trans_record[i][
                    -2] = cash_have * adjust2  # turn back to the cash that can be determine later in the other process
            else:
                dp_record[i][-1] = dp_record[i - 1][-1]
                trans_record[i][-1] = use_cash
                trans_record[i][-2] = trans_record[i - 1][-2]

            for j in range(4):
                stock_have = cash_have / priceMat_masked[i][j] * adjust1
                if stock_have > dp_record[i - 1][j]:
                    dp_record[i][j] = stock_have
                    trans_record[i][j] = sell_idx  # change from which stock or cash
                else:
                    dp_record[i][j] = dp_record[i - 1][j]
                    trans_record[i][j] = j

        if trans_record[-1][-2] > last_record:
            action_matrix = []
            last_record = trans_record[-1][-2]
            # print(mask,'----------------')
            # print(priceMat_masked[mask-4:mask+2])

            sell_from = trans_record[-1][-1]
            buy_to = -1
            for i in range(days - 1, -1, -1):
                # print(i, trans_record[i], dp_record[i], end='\n')
                if i == 0:
                    sell_from = -1
                if not (sell_from ^ use_cash):  # let 4 to -1
                    sell_from = -1
                if not (buy_to ^ use_cash):
                    buy_to = -1

                if (sell_from ^ buy_to):  # if transfer then add to actionMRX
                    action_matrix.append([i, sell_from, buy_to, trans_record[i][-2]])
                    # print([i, sell_from, buy_to, trans_record[i][-2]])

                buy_to = sell_from
                sell_from = trans_record[i - 1][buy_to]

            # actionMat = action_matrix
    actionMat = action_matrix[::-1]


    return actionMat


def myAction04(priceMat, transFeeRate, K):
    days, types = priceMat.shape
    cash = 1000
    use_cash = 4
    adjust1 = (1 - transFeeRate)
    adjust2 = 100.0 / 99.0
    # dp_record = np.zeros((days, 5), dtype=float)
    dp_record = [[0 for j in range(5)] for i in range(days)]
    trans_record = [[0 for j in range(6)] for i in range(days)]
    action_matrix = []

    # init
    dp_record[0] = [cash / priceMat[0][0] * adjust1, cash / priceMat[0][1] * adjust1, cash / priceMat[0][2] * adjust1,
                    cash / priceMat[0][3] * adjust1, cash]
    trans_record[0] = [0, 1, 2, 3, 1000.0, 4]
    cash_have = cash

    for i in range(1, days):
        sell_idx = -1  # no stock sold
        for j in range(4):
            cash_new = dp_record[i - 1][j] * priceMat[i][j] * adjust1  # yesterday stock equal how much cash today
            if cash_new > cash_have:
                cash_have = cash_new
                sell_idx = j

        if cash_have > dp_record[i - 1][-1]:
            dp_record[i][-1] = cash_have
            trans_record[i][-1] = sell_idx
            trans_record[i][
                -2] = cash_have * adjust2  # turn back to the cash that can be determine later in the other process
        else:
            dp_record[i][-1] = dp_record[i - 1][-1]
            trans_record[i][-1] = use_cash
            trans_record[i][-2] = trans_record[i - 1][-2]

        for j in range(4):
            stock_have = cash_have / priceMat[i][j] * adjust1
            if stock_have > dp_record[i - 1][j]:
                dp_record[i][j] = stock_have
                trans_record[i][j] = sell_idx  # change from which stock or cash
            else:
                dp_record[i][j] = dp_record[i - 1][j]
                trans_record[i][j] = j

    sell_from = trans_record[-1][-1]
    buy_to = -1
    for i in range(days - 1, -1, -1):
        # print(i, trans_record[i], dp_record[i], end='\n')
        if i == 0:
            sell_from = -1
        if not (sell_from ^ use_cash):  # let 4 to -1
            sell_from = -1
        if not (buy_to ^ use_cash):
            buy_to = -1

        if (sell_from ^ buy_to):  # if transfer then add to actionMRX
            action_matrix.append([i, sell_from, buy_to, trans_record[i][-2]])
            # print(i,[i, sell_from, buy_to, trans_record[i][-2]])

        buy_to = sell_from
        sell_from = trans_record[i - 1][buy_to]

    # actionMat = action_matrix
    actionMat = action_matrix[::-1]

    #TODO: deal with cash holding
    r_list = []  # [[buy day, sell day, return rate],[]...]
    r = []
    last_cash = 0
    for data in actionMat:
        if data[1] == -1:
            # print('buy stock',data)
            last_day = data[0]
            last_cash = data[3]
            r.append(data)

        if data[2] == -1:
            # print('sell stock',data)
            r.append(data)
            r.append(data[3] / last_cash)
            r.append(data[0] - last_day)
            r_list.append(r)
            # init
            r = []
    for i in range(len(r_list)):
        end = 0
        c = i
        while not end:
            r_list[c]

