import pandas as pd

def PivotPoint(args): #data: pd.DataFrame, method = "classic"):
    method = args["method"]
    data = []
    high = args.get("high", None)
    if high is not None:
        data.append(high)
    low = args.get("low", None)
    if low is not None:
        data.append(low)
    close = args.get("close", None)
    if close is not None:
        data.append(close)
    open = args.get("open", None)
    if open is not None:
        data.append(open)
    data = pd.concat(data, axis = 1)

    if method == "classic":
        '''
        Formula:
        previous high, low, close
        - pivot = (h + l + c) / 3  # variants duplicate close or add open
        - support1 = 2.0 * pivot - high
        - support2 = pivot - (high - low)
        - resistance1 = 2.0 * pivot - low
        - resistance2 = pivot + (high - low)

        See:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        - https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)
        '''
        h = data.high
        l = data.low
        c = data.close
        p = (h + l + c) / 3.0
        s1 = 2.0 * p - h
        s2 = p - (h - l)
        s3 = l - 2 * (h - p) #p - (h - l) * 2.0
        r1 = 2.0 * p - l
        r2 = p + (h - l)
        r3 = h + 2 * (p - l) #p + (h - l) * 2.0

        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        })
        # out = out.shift(1)
        return out 
    elif method == 'fibonacci':
        '''Formula:
        previous high, low, close
        - pivot = (h + l + c) / 3  # variants duplicate close or add open
        - support1 = p - level1 * (high - low)  # level1 0.382
        - support2 = p - level2 * (high - low)  # level2 0.618
        - support3 = p - level3 * (high - low)  # level3 1.000
        - resistance1 = p + level1 * (high - low)  # level1 0.382
        - resistance2 = p + level2 * (high - low)  # level2 0.618
        - resistance3 = p + level3 * (high - low)  # level3 1.000

        See:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        https://tradingliteracy.com/fibonacci-pivot-points/
        '''
        h = data.high
        l = data.low
        c = data.close

        p = (h + l + c) / 3.0
        s1 = p - 0.382 * (h - l)
        s2 = p - 0.618 * (h - l)
        s3 = p - (h - l) * 1.0
        r1 = p + 0.382 * (h - l)
        r2 = p + 0.618 * (h - l)
        r3 = p + (h - l)

        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        })
        # out = out.shift(1)
        return out 
    elif method == "demark":
        '''Formula:
        - if close < open x = high + (2 x low) + close

        - if close > open x = (2 x high) + low + close

        - if Close == open x = high + low + (2 x close)

        - p = x / 4

        - support1 = x / 2 - high
        - resistance1 = x / 2 - low

        See:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        '''
        def getx(v):
            if v.close < v.open:
                return v.high + (2 * v.low) + v.close
            elif v.close > v.open:
                return 2 * v.high + v.low + v.close
            else:
                return v.high + v.low + 2 * v.close
        x = data.apply(getx, axis = 1)
        p = x / 4
        s1 = x / 2 - data.high
        r1 = x / 2 - data.low

        out = pd.DataFrame({"p": p, "r1": r1, "s1": s1})
        # out = out.shift(1)
        out['r2'] = None
        out['r3'] = None
        out['s2'] = None
        out['s3'] = None
        return out
    elif method == "woodie":
        '''
        PP = (H + L + 2*C) / 4
        R1 = (2*PP) -L
        R2 = PP + H - L
        R3 = H + 2*(PP-L)
        S1 = (2*PP) - H
        S2 = PP - R1 -S1
        S3 = L - 2*(H - PP)
        Where,

        H is the previous day's price high
        L is the previous day's price low
        C is the previous day's closing price
        PP is the pivot point
        R1, R2, R3 are resistance lines 1, 2, and 3, respectively
        S1, S2, S3 are support lines 1, 2, and 3, respectively        
        https://tradeveda.com/woodie-pivot-points-in-technical-analysis-trading-guide/
        https://www.sohomarkets.com/upload/file/20220510/20220510140897779777.pdf
        '''
        h = data.high
        l = data.low
        c = data.close
        p = (h + l + 2 * c) / 4
        r1 = 2 * p - l
        r2 = p + h - l
        r3 = h + 2 * (p - l)
        s1 = 2 * p - h
        s2 = p - h + l
        s3 = l - 2 * (h - p)
        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        })
        # out = out.shift(1)
        return out 
    elif method == "camarilla":
        '''
        Resistance 4 or R4 = (H-L)X1.1/2+C
        Resistance 3 or R3 = (H-L)X1.1/4+C
        The Resistance 2 or R2 = (H-L)X1.1/6+C
        Resistance 1 or R1 = (H-L)X1.1/12+C
        PIVOT POINT = (H+L+C)/3
        Support 1 or S1 = C-(H-L)X1.1/12
        Support 2 or S2 = C-(H-L)X1.1/6
        The Support 3 or S3 = C-(H-L)X1.1/4
        Support 4 or S4 = C-(H-L)X1.1/2
        Here O, H, L, and C represent the open, high, low and close values of the previous trading day.        
        https://www.stockmaniacs.net/freebies/free-tools/camarilla-calculator/
        '''
        h = data.high
        l = data.low
        c = data.close
        p = (h + l + c) / 3
        r1 = c + (h -l) * 1.1 / 12 #1.0833 #
        r2 = c + (h -l) * 1.1 / 6 #1.1666 #
        r3 = c + (h -l) * 1.1 / 4 #1.2500 #
        r4 = c + (h -l) * 1.1 / 2 #1.5000 #
        s1 = c - (h -l) * 1.1 / 12 #1.0833 #
        s2 = c - (h -l) * 1.1 / 6 #1.1666 #
        s3 = c - (h -l) * 1.1 / 4 #1.2500 #
        s4 = c - (h -l) * 1.1 / 2 #1.5000 #
        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            'r4': r4,
        })
        # out = out.shift(1)
        return out         
    
def PivotPoint_v1(data: pd.DataFrame, method = "classic"):
    if method == "classic":
        '''
        Formula:
        previous high, low, close
        - pivot = (h + l + c) / 3  # variants duplicate close or add open
        - support1 = 2.0 * pivot - high
        - support2 = pivot - (high - low)
        - resistance1 = 2.0 * pivot - low
        - resistance2 = pivot + (high - low)

        See:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        - https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)
        '''
        h = data.high
        l = data.low
        c = data.close
        p = (h + l + c) / 3.0
        s1 = 2.0 * p - h
        s2 = p - (h - l)
        s3 = l - 2 * (h - p) #p - (h - l) * 2.0
        r1 = 2.0 * p - l
        r2 = p + (h - l)
        r3 = h + 2 * (p - l) #p + (h - l) * 2.0

        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        })
        # out = out.shift(1)
        return out 
    elif method == 'fibonacci':
        '''Formula:
        previous high, low, close
        - pivot = (h + l + c) / 3  # variants duplicate close or add open
        - support1 = p - level1 * (high - low)  # level1 0.382
        - support2 = p - level2 * (high - low)  # level2 0.618
        - support3 = p - level3 * (high - low)  # level3 1.000
        - resistance1 = p + level1 * (high - low)  # level1 0.382
        - resistance2 = p + level2 * (high - low)  # level2 0.618
        - resistance3 = p + level3 * (high - low)  # level3 1.000

        See:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        https://tradingliteracy.com/fibonacci-pivot-points/
        '''
        h = data.high
        l = data.low
        c = data.close

        p = (h + l + c) / 3.0
        s1 = p - 0.382 * (h - l)
        s2 = p - 0.618 * (h - l)
        s3 = p - (h - l) * 1.0
        r1 = p + 0.382 * (h - l)
        r2 = p + 0.618 * (h - l)
        r3 = p + (h - l)

        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        })
        # out = out.shift(1)
        return out 
    elif method == "demark":
        '''Formula:
        - if close < open x = high + (2 x low) + close

        - if close > open x = (2 x high) + low + close

        - if Close == open x = high + low + (2 x close)

        - p = x / 4

        - support1 = x / 2 - high
        - resistance1 = x / 2 - low

        See:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        '''
        def getx(v):
            if v.close < v.open:
                return v.high + (2 * v.low) + v.close
            elif v.close > v.open:
                return 2 * v.high + v.low + v.close
            else:
                return v.high + v.low + 2 * v.close
        x = data.apply(getx, axis = 1)
        p = x / 4
        s1 = x / 2 - data.high
        r1 = x / 2 - data.low

        out = pd.DataFrame({"p": p, "r1": r1, "s1": s1})
        # out = out.shift(1)
        out['r2'] = None
        out['r3'] = None
        out['s2'] = None
        out['s3'] = None
        return out
    elif method == "woodie":
        '''
        PP = (H + L + 2*C) / 4
        R1 = (2*PP) -L
        R2 = PP + H - L
        R3 = H + 2*(PP-L)
        S1 = (2*PP) - H
        S2 = PP - R1 -S1
        S3 = L - 2*(H - PP)
        Where,

        H is the previous day's price high
        L is the previous day's price low
        C is the previous day's closing price
        PP is the pivot point
        R1, R2, R3 are resistance lines 1, 2, and 3, respectively
        S1, S2, S3 are support lines 1, 2, and 3, respectively        
        https://tradeveda.com/woodie-pivot-points-in-technical-analysis-trading-guide/
        https://www.sohomarkets.com/upload/file/20220510/20220510140897779777.pdf
        '''
        h = data.high
        l = data.low
        c = data.close
        p = (h + l + 2 * c) / 4
        r1 = 2 * p - l
        r2 = p + h - l
        r3 = h + 2 * (p - l)
        s1 = 2 * p - h
        s2 = p - h + l
        s3 = l - 2 * (h - p)
        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        })
        # out = out.shift(1)
        return out 
    elif method == "camarilla":
        '''
        Resistance 4 or R4 = (H-L)X1.1/2+C
        Resistance 3 or R3 = (H-L)X1.1/4+C
        The Resistance 2 or R2 = (H-L)X1.1/6+C
        Resistance 1 or R1 = (H-L)X1.1/12+C
        PIVOT POINT = (H+L+C)/3
        Support 1 or S1 = C-(H-L)X1.1/12
        Support 2 or S2 = C-(H-L)X1.1/6
        The Support 3 or S3 = C-(H-L)X1.1/4
        Support 4 or S4 = C-(H-L)X1.1/2
        Here O, H, L, and C represent the open, high, low and close values of the previous trading day.        
        https://www.stockmaniacs.net/freebies/free-tools/camarilla-calculator/
        '''
        h = data.high
        l = data.low
        c = data.close
        p = (h + l + c) / 3
        r1 = c + (h -l) * 1.1 / 12 #1.0833 #
        r2 = c + (h -l) * 1.1 / 6 #1.1666 #
        r3 = c + (h -l) * 1.1 / 4 #1.2500 #
        r4 = c + (h -l) * 1.1 / 2 #1.5000 #
        s1 = c - (h -l) * 1.1 / 12 #1.0833 #
        s2 = c - (h -l) * 1.1 / 6 #1.1666 #
        s3 = c - (h -l) * 1.1 / 4 #1.2500 #
        s4 = c - (h -l) * 1.1 / 2 #1.5000 #
        out = pd.DataFrame({
            'p': p,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            'r4': r4,
        })
        # out = out.shift(1)
        return out         