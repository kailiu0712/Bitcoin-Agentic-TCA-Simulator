**Feed integrity and reconstruction health**

| metric                                                  |           normal |           stress |
|:--------------------------------------------------------|-----------------:|-----------------:|
| raw feed rows processed                                 |      8.56423e+07 |      9.18206e+07 |
| packets (distinct receipt timestamps)                   |      6.17456e+07 |      6.6356e+07  |
| measured hours                                          |    167.954       |    191.956       |
| aggressive orders (trade bursts)                        | 187779           | 210208           |
| individual fills                                        | 396688           | 415105           |
| snapshot blocks seen                                    |     65           |     42           |
| levels evicted by the depth window                      |      2.34136e+07 |      2.48637e+07 |
| levels dropped by the crossed-book repair               |      0           |      6           |
| crossed packets after repair (must be 0)                |      0           |      0           |
| orphan deletes (order already evicted)                  |    191           |    208           |
| idempotent re-notifications (scroll-in)                 |   6717           |  35944           |
| prices off the 0.1 USD tick grid                        |      0           |      0           |
| receipt-clock reversals                                 |      0           |      0           |
| level-capacity overflows                                |      0           |      0           |
| order-slot exhaustions                                  |      0           |      0           |
| pre-trade state lookups without an old-enough entry     |    300           |    198           |
| aggressor-side agreement (exchange-aligned)             |      0.99206     |      0.98752     |
| aggressor-side agreement (receipt-aligned)              |      0.77669     |      0.73338     |
| ingest throughput (rows/s)                              | 590476           | 544055           |
| ingest wall time (s)                                    |    145           |    168.8         |
| snapshot checks                                         |     64           |     42           |
| median |best-bid error| (ticks)                         |      0           |      0           |
| median |best-ask error| (ticks)                         |      0           |      0           |
| share of snapshots with best quotes exact               |      0.78125     |      0.738095    |
| median reconstructed/published bid depth                |      1.0064      |      0.998405    |
| median reconstructed/published ask depth                |      0.998603    |      1.00041     |
| share of snapshot mismatches that are whole-book shifts |      0.9375      |      0.928571    |
