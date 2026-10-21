## Ticks

To better undestand the tick system I have decided to fetch the mini S&P 500 data from Databento. The fetched data has many symbols and this is what I understand all of those fields mean:

- `ts_recv` - timestamp in UTC when Databento received the trade message. A trade message desribes the actual execution of a trade the moment the an exchanges matching engine makes a match. It is a permanent record of the transation being committed. While limit orders represent the intention to sell or buy at a given price, a trade message actually represents an executed transaction.

- `ts_event` - timestamp when CME's (which stands for the Chicago Mercantile Exchange, world's largest and most dominant financial derivatives exchange) matching engine executed the trade in UTC.

- `rtype` - a record type. Because Databento can store multiple types of data it basically tell the parser which kind of record it can expect. We can have:
  - **MBO (Market By Order)** A single order book event. The row represents a specific trader adding, modifying, or canceling an individual limit order in the queue.

  - **MBP-1 / MBP-10 (Market By Price)** An update to the aggregated order book. The row shows the total volume resting at the best price level (MBP-1) or the top 10 price levels (MBP-10).

  - **Trade** A completed transaction. The row represents a matched buy and sell, as discussed previously.

  - **OHLCV (Open, High, Low, Close, Volume)** Time-aggregated summary bars (e.g., a 1-second, 1-minute, or daily summary of price action).

- `publisher_id` - the id of the publisher. Because we deal with the S&P 500 the publisher is always CME Globex.

- `instrument_id` - the id of the instrument being traded, right now it is constant because we analyse the E-mini S&P 500. The E-mini was introduced because the original contract was to high and the change in one tick becase a big difference. E-mini is worth 1/5-th what the "original" contract was worth. The original SP500 was discontinued.

- `action` - becase we fetched trading data it will always be it.

- `side` - categorical feature. **B** - buyer hit the ask, **A** - seller hit the bid. **N** is unknown.

- `depth` - book level the trade happened at (always 0 for trades). When a bigger order is processed than what is available at the current price, the order will be split into suborders, each matched at depth 0 until the contract is fulfilled.

- `price` - trade price in index points (1 point = $50 per contract). For the S&P500 data the value always moves in 0.25 incerements and each tick is worth \$50 (so the 0.25 increment is worth $12.5). This does not apply to the individual stocks. When we look at commonly listed symbols over \$1.00, then can only move in \$0.01 increments (governed by the US Stock Exchange). This is the ticksize for the individual stocks, the price cannot move in smaller increments. So then how the connecton from the actual SP500 companies is made? It is made using a mathematical expiration rule and algorithmic arbitrage. So first thing is the actual calculation of the SP500. It is a market-capitalization-weighted average of 500 large US companies. It is represented by:

  $$
      \text{Index} = \frac{1}{D}\sum_{i=1}^{500} P_i Q_i,
  $$

  where:
  - $P_i$ - the price of $i$-th company,
  - $Q_i$ - the number of publicly tradable number of shares of the company,
  - $D$ - proprietary number maintained by S&P Dow Jones Indices (currently in the billions). It scales the multi-trillion-dollar market cap sum down to a readable reporting level

  A computer continuously calculates these 500 changing market caps and spits out the spot value of the S&P 500 Index. Futures contracts have an expiration date (for the ES - official tick of E-mini S&P500, this is the third Friday of the contract month). The SP500 index is **cash-settled**, which means that on the exact moment of the contract expiration it is instantly converted to cash based on the current value of the calculated SP500 index. Because of this it is traded extremely closely to the actual price of the index. The other thing keeping the price of the SP500 index very closely to the actual trading value of the underlying companies is the HFT arbitrage. If the index is undervalued, the automatic algorithms will sell their shares of the listed companies, driving their prices down and buy the index, giving them profit. The mirrored situation happens when the stock is overvalued. There is a little spread between the SP500 value, it trades with a slight premium. This gap is called **fair-value**. Because a future is an agreement to buy later, holding the future is mathematically equivalent to holding your cash in a bank account (earning interest) minus the stock dividends you are missing out on by not owning the actual shares.

- `size` - how many contracts were traded. The actual value of the traded asset is size $\times$ price $\times$ \$50.

- `flags` - some additional information about the packets.

- `ts_in_delta` - tells us how long it took for the packet to travel from the CME server to Databento.

- `sequence` - CME's message sequence number.

- `symbol` - the indetifier of the traded symbol, here it is contants and equal to ES.v.0.

Okay I think that right now we know what the data is about. We can go over to analysing the ticks and processing the data.
