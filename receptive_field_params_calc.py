def calc(NL_PG=10,NL_R=10,N_r=3,W_max=20,is_offline=True):
    r_dilation = []

    pg_dilation1 = []
    pg_dilation2 = []
    future_vec = []

    if is_offline: # RR-MS-TCN
        for i in range(NL_PG):
            pg_dilation1.append(2**i)
            pg_dilation2.append(2**(NL_PG-i-1))
            future_vec.append(max(2**i,2**(NL_PG-i-1)))

        PG_future = sum(future_vec)
        PG_rceptive = 2*PG_future + 1
        for i in range(NL_R):
            r_dilation.append(2**i)
        R_future = sum(r_dilation)
        R_receptive = 2*R_future +1

        total_receptive = PG_rceptive + N_r*(R_receptive - 1)
        total_future = PG_future + N_r * R_future
        # total_receptive1 = total_future*2 +1
        print(total_future)
        return total_future, total_receptive
    else:   # BF-MS-TCN
        for i in range(NL_PG):
            delta_1 = min(2**i,W_max)
            delta_2 = min(2**(NL_PG-i-1),W_max)
            pg_dilation1.append(delta_1)
            pg_dilation2.append(delta_2)
            future_vec.append(max(delta_1, delta_2))
        PG_future = sum(future_vec)

        for i in range(NL_R):
            r_dilation.append(min(2**i,W_max))
        R_future = sum(r_dilation)
        total_future = PG_future + (N_r * R_future)
        print(total_future)
        return total_future, "not impementet yet"



BF_future_win,_ = calc(NL_PG=10,NL_R=10,N_r=3,W_max=2**10,is_offline=False) # BF-MS-TCN
RR_future_win,_ = calc(NL_PG=10,NL_R=10,N_r=3,W_max=10,is_offline=True)   # RR-MS-TCN

fps = 30

assert BF_future_win == RR_future_win, f"BF_future_win: {BF_future_win} != RR_future_win: {RR_future_win}"

print(f"BF_future_win (sec): {BF_future_win/fps}\nRR_future_win (sec): {RR_future_win/fps}")