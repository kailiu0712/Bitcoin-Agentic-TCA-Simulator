**Interarrival statistics by event stream**

| regime   | stream             |   events |   mean_interarrival_ms |   median_interarrival_ms |   p05_interarrival_ms |   p95_interarrival_ms |   share_zero_gap |   events_per_second |
|:---------|:-------------------|---------:|-----------------------:|-------------------------:|----------------------:|----------------------:|-----------------:|--------------------:|
| normal   | aggressive orders  |   161702 |               3740.23  |             1760         |             0.030857  |             14221.5   |       0.00336423 |            0.267363 |
| normal   | fills              |   396688 |               1742.43  |                0         |             0         |              9509.93  |       0.528006   |            0.573912 |
| normal   | top-of-book events |  4490391 |                134.687 |                1.70081   |             0.022313  |               707.966 |       0          |            7.4246   |
| stress   | aggressive orders  |   195933 |               3527.61  |             1602.22      |             0.0295256 |             13504.1   |       0.00166384 |            0.283478 |
| stress   | fills              |   415105 |               1873.2   |                0.0188265 |             0         |              9903.91  |       0.494462   |            0.533847 |
| stress   | top-of-book events |  6199245 |                111.497 |                1.53917   |             0.021765  |               512.834 |       0          |            8.96882  |
