import numpy as np

def generate_trade_report_from_earnings()

def generate_trade_report(data: dict) -> dict:
    date: list[str] = []
    sec: list[str]  = []
    decision = []
    confidence = []
    count = 0
    
    for result in data:
        count += 1
        try:
            date.append(result['date'])
            sec.append(result['security'])
            response = result['response']
            try:
                decision.append(response['decision'])
            except:
                try:
                    #response = result['response']

                    isBuy = False
                    isSell = False
                    isHold = False

                    if response.find("BUY") != -1:
                        isBuy = True
                    if response.find("SELL") != -1:
                        isSell = True
                    if response.find("HOLD") != -1:
                        isHold = True

                    if isBuy and not(isSell or isHold):
                        decision.append('BUY')
                    elif isSell and not(isBuy or isHold):
                        decision.append('SELL')
                    elif isHold and not(isBuy or isSell):
                        decision.append('HOLD')
                    else:
                        decision.append('Missing')
                except:
                    decision.append('Missing')
            try:
                if 'confidence score' in response.keys():
                    confidence.append(response['confidence score'])
                elif 'confidence' in response.keys():
                    confidence.append(response['confidence'])
                else:
                    confidence.append(np.nan)
            except:
                    confidence.append(np.nan) 
                
        except:
            print(f'Missing date {count}')
        
        
        
    return {'Date': date, 'Security': sec, 'Decision': decision, 'Confidence': confidence}
        