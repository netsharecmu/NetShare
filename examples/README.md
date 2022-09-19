# PCAP
## w/o differential privacy
```Bash
python3 driver.py
```

## w/ differential privacy (two-step)
- Step 1: pretrain public model
    ```Bash
    python3 driver_public.py
    ```

- Step 2: train private model/data
    - Determine DP parameters by modifying
        ```Json
        "model": {
            "config": {
                "dp_noise_multiplier": <noise_multiplier>,
                "dp_l2_norm_clip": 1.0,
            }
        }
        ```
    - Specify public model location from step 1:
        ```Json
        "model_manager": {
            "pretrain_dir": <public model path>
        }
        ```
    - Run training and generation
        ```Bash
        python3 driver_private.py
        ```