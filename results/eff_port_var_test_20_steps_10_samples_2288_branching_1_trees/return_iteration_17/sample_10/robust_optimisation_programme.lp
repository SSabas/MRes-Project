\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  0.945227 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.16316 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.16709 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.751851 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.22739 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.973418 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.851554 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.44736 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.07879 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.978235 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.915671 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.789457 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.672013 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.26433 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.12301 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.13375 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.16703 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.55953 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.926422 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.5286 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.18812 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.947113 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.843307 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.872038 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.12255 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.88914 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.09096 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.29921 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.09981 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.5123 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.797819 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.800593 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.809148 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.641543 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.05838 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.25028 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.77379 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.978127 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.972111 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.28143 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.14882 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.826517 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.38294 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.896472 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.796438 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.40064 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.993925 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.17344 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.931329 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.724935 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.470048 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.41926 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.07825 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.988597 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.1922 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.58872 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.697423 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.6453 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.907999 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.716482 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.850302 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.964996 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.971704 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.924366 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.1171 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.21426 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.11453 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.39992 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.935628 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.687964 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.911229 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.670058 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.3959 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.54793 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.927654 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.13974 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.976469 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.39182 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.14596 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.749309 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.3889 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.831333 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.779312 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.78352 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.941867 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.18292 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.800435 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.590196 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.533586 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.55546 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.00935 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.950708 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.14897 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 2.0461 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.566933 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.40795 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.01602 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.794959 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.829248 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.01007 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.20541 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.791007 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.979894 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.65649 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.33569 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.67787 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.09817 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.723441 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.800454 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.695375 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.37591 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.76905 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.943692 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.09466 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.972918 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.17494 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.10749 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.792827 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.2349 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.88784 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.830746 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.53428 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.0886 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.12503 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.929096 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.808918 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.641104 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.21669 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.959478 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.05094 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.08377 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.4048 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.774704 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.27949 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.03437 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.776847 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.924795 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.972088 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.968044 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.785472 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.08696 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.50635 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.25025 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.34601 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.12186 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.83814 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.870405 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.750035 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.19788 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.52476 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.813835 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.11174 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.943549 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.4163 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.98572 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.1399 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.26929 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.941536 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.13673 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.07761 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.676336 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.13657 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.931036 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.639591 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.6676 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.76494 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.44079 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.583136 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.00508 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.76993 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.620734 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.44667 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.772378 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.852601 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.67771 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.921192 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.46268 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.3705 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.906817 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.859968 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.912666 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.42919 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.534628 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.646187 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.14326 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.873381 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.64174 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.918285 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.6049 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.907494 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.985174 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.11543 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.14702 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.910993 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.12405 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.864968 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.808955 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.18072 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.900931 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.97775 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.83396 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.701339 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.595916 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.15992 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.851783 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.03075 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.962 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.2108 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.729964 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13463 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.02496 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.820235 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.18372 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.964273 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.978265 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.895209 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.27959 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.20179 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.14826 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.41701 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.27884 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.01694 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.951195 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.84363 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.04937 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.60253 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.06321 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.2342 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.925736 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.36707 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.08075 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.962733 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.31854 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.804308 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.989851 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.40632 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.923686 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.14336 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.849451 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.671912 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.634783 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.55395 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.1324 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.835285 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.11893 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.73813 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.596608 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.26064 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.90811 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.890709 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.708773 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.807083 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.24405 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.878993 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.866186 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.05533 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.990256 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.3172 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.805292 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.771328 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.03052 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.859992 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.53074 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.29574 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.21883 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.14626 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.925638 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.6742 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.958261 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.11945 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.17502 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.786369 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.26758 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.43534 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.820912 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.29009 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.1641 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.620424 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.631899 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 2.20477 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.39307 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.541899 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.06038 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.97316 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.636212 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.83097 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.73864 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.622764 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.68494 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.994409 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.47431 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.32801 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.07987 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.928874 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.8507 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.31992 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.527241 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.529115 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.08829 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.87638 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.7182 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.859163 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.48329 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.10457 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.950534 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.1848 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.12115 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.896161 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.19223 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.847827 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.876574 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.43483 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.0122 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.1204 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.928417 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.743796 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.745331 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.17644 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.85919 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.952682 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.01618 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.35789 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.704619 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.05661 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.06764 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.825185 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.959944 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.981085 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.09301 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.786782 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.12876 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.38581 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.21104 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.30022 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.1809 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.905895 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.917018 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.789108 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17198 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.3712 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.971216 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.0874 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.992644 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.14672 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08859 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.864485 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.25751 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.939946 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.906448 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.16241 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.90947 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.945739 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.914717 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.719916 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.746849 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.17247 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.04751 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.9422 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.06056 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.46877 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.844925 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.13946 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.03243 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.900478 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.784982 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.857821 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.14204 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.824014 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.992411 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.994665 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.03179 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.02826 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.946985 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.845732 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.911119 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.884846 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.2153 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.16809 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.924133 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.14164 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.374001446088371 z_k1_s0000
                                 - 0.858003317496852 z_k1_s0001
                                 - 0.741004081660633 z_k1_s0002
                                 - 0.528002041536524 z_k1_s0003
                                 - 0.815000109861064 z_k1_s0004
                                 - 1.42800491315347 z_k1_s0005
                                 - 0.656001989007097 z_k1_s0006
                                 - 1.29100620826118 z_k1_s0007
                                 - 0.932995562048582 z_k1_s0010
                                 - 0.661997654645521 z_k1_s0011
                                 - 0.562998429051014 z_k1_s0012
                                 - 0.472998397567346 z_k1_s0013
                                 - 0.635997885265062 z_k1_s0014
                                 - 2.01599342389641 z_k1_s0015
                                 - 0.974000280794549 z_k1_s0016
                                 - 0.484998446711698 z_k1_s0017
                                 - 1.57800146913497 z_k1_s0020
                                 - 0.410001102354035 z_k1_s0021
                                 - 0.91400342854652 z_k1_s0022
                                 - 0.459001069919396 z_k1_s0023
                                 - 1.28100549467997 z_k1_s0024
                                 - 1.38900077583819 z_k1_s0025
                                 - 0.409000708429375 z_k1_s0026
                                 - 0.72099972734791 z_k1_s0027
                                 - 1.40799827955917 z_k1_s0030
                                 - 1.69599818383291 z_k1_s0031
                                 - 0.762000404258292 z_k1_s0032
                                 - 0.85600109317062 z_k1_s0033
                                 - 1.43999952136869 z_k1_s0034
                                 - 1.19299592440781 z_k1_s0035
                                 - 2.64000663718387 z_k1_s0036
                                 - 1.27700622916523 z_k1_s0037
                                 - 1.16800257648531 z_k1_s0040
                                 - 0.59300044558066 z_k1_s0041
                                 - 1.13400429427393 z_k1_s0042
                                 - 0.495000644426562 z_k1_s0043
                                 - 1.33700470677948 z_k1_s0044
                                 - 0.550000532829372 z_k1_s0045
                                 - 1.31899979689839 z_k1_s0046
                                 - 1.64800160254619 z_k1_s0047
                                 - 0.341000000165175 z_k1_s0050
                                 - 0.561999349948827 z_k1_s0051
                                 - 0.548999096461846 z_k1_s0052
                                 - 1.52299716050742 z_k1_s0053
                                 - 1.43500154858944 z_k1_s0054
                                 - 0.475999580533039 z_k1_s0055
                                 - 0.711000293301581 z_k1_s0056
                                 - 1.48799729554165 z_k1_s0057
                                 - 0.360000517123873 z_k1_s0060
                                 - 0.978999318982076 z_k1_s0061
                                 - 0.326000314339803 z_k1_s0062
                                 - 0.7050001297834 z_k1_s0063
                                 - 0.293000117519971 z_k1_s0064
                                 - 0.562000091898246 z_k1_s0065
                                 - 0.469000352224958 z_k1_s0066
                                 - 0.381000234872464 z_k1_s0067
                                 - 0.569999224384257 z_k1_s0070
                                 - 0.57999928904004 z_k1_s0071
                                 - 0.651999011062449 z_k1_s0072
                                 - 0.670999133908437 z_k1_s0073
                                 - 0.522999664001307 z_k1_s0074
                                 - 2.07499482859424 z_k1_s0075
                                 - 0.723999476584088 z_k1_s0076
                                 - 1.63999573356383 z_k1_s0077
                                 - 0.631998684520027 z_k1_s0100
                                 - 0.492997430837241 z_k1_s0101
                                 - 0.703996870724932 z_k1_s0102
                                 - 0.270998947387846 z_k1_s0103
                                 - 0.248999026071516 z_k1_s0104
                                 - 0.195999104464421 z_k1_s0105
                                 - 0.37099914555084 z_k1_s0106
                                 - 0.140999606872416 z_k1_s0107
                                 - 0.212000068399848 z_k1_s0110
                                 - 0.132999870060538 z_k1_s0111
                                 - 0.108999859362446 z_k1_s0112
                                 - 0.203000279712745 z_k1_s0113
                                 - 0.387999416053601 z_k1_s0114
                                 - 0.198000049109314 z_k1_s0115
                                 - 0.102999895837865 z_k1_s0116
                                 - 0.219999184567191 z_k1_s0117
                                 - 1.53799959191203 z_k1_s0120
                                 - 0.397998893831321 z_k1_s0121
                                 - 0.829001183736411 z_k1_s0122
                                 - 0.643998659943217 z_k1_s0123
                                 - 0.566998634154739 z_k1_s0124
                                 - 0.610998554862635 z_k1_s0125
                                 - 0.542998737241217 z_k1_s0126
                                 - 1.45199935795649 z_k1_s0127
                                 - 0.142999851356696 z_k1_s0130
                                 - 0.271999545363966 z_k1_s0131
                                 - 0.295000561294374 z_k1_s0132
                                 - 0.161999789474064 z_k1_s0133
                                 - 0.287999518620669 z_k1_s0134
                                 - 0.414999155720914 z_k1_s0135
                                 - 0.38199906025412 z_k1_s0136
                                 - 0.45299999595464 z_k1_s0137
                                 - 0.857001671958242 z_k1_s0140
                                 - 0.943001347284059 z_k1_s0141
                                 - 0.23400025636144 z_k1_s0142
                                 - 0.545998782975906 z_k1_s0143
                                 - 0.613002382056994 z_k1_s0144
                                 - 0.357000076946163 z_k1_s0145
                                 - 0.360000150048263 z_k1_s0146
                                 - 0.628000478566509 z_k1_s0147
                                 - 0.319998673404347 z_k1_s0150
                                 - 0.129999807869841 z_k1_s0151
                                 - 0.138999780759319 z_k1_s0152
                                 - 0.408999620242371 z_k1_s0153
                                 - 0.217000089800398 z_k1_s0154
                                 - 0.155999638884063 z_k1_s0155
                                 - 0.159999663101538 z_k1_s0156
                                 - 0.100999877092671 z_k1_s0157
                                 - 0.336999760554206 z_k1_s0160
                                 - 0.218000265981404 z_k1_s0161
                                 - 0.272000551022539 z_k1_s0162
                                 - 0.0850000228946318 z_k1_s0163
                                 - 0.100999897225463 z_k1_s0164
                                 - 0.0679999884557231 z_k1_s0165
                                 - 0.286999220145911 z_k1_s0166
                                 - 0.124999858021621 z_k1_s0167
                                 - 0.307000677532572 z_k1_s0170
                                 - 0.20600055616607 z_k1_s0171
                                 - 0.533002800863964 z_k1_s0172
                                 - 0.539000024005904 z_k1_s0173
                                 - 0.235000555650427 z_k1_s0174
                                 - 0.832003500931939 z_k1_s0175
                                 - 0.392001493371411 z_k1_s0176
                                 - 0.203000321091271 z_k1_s0177
                                 - 0.205999668506376 z_k1_s1000
                                 - 0.12099978171495 z_k1_s1001
                                 - 0.18200014322334 z_k1_s1002
                                 - 0.139000336175683 z_k1_s1003
                                 - 0.124999297929476 z_k1_s1004
                                 - 0.0659998478263927 z_k1_s1005
                                 - 0.108999776273844 z_k1_s1006
                                 - 0.265999088712937 z_k1_s1007
                                 - 0.340999223767004 z_k1_s1010
                                 - 0.133000275881783 z_k1_s1011
                                 - 0.170999263564082 z_k1_s1012
                                 - 0.123999798742414 z_k1_s1013
                                 - 0.199999429004298 z_k1_s1014
                                 - 0.116999837822232 z_k1_s1015
                                 - 0.102999788682078 z_k1_s1016
                                 - 0.0839999129415545 z_k1_s1017
                                 - 0.0160000065900883 z_k1_s1020
                                 - 0.0579999162266178 z_k1_s1021
                                 - 0.0449999350034104 z_k1_s1022
                                 - 0.0559999191153552 z_k1_s1023
                                 - 0.0200000008126136 z_k1_s1024
                                 - 0.0129999812232074 z_k1_s1025
                                 - 0.037999945113991 z_k1_s1026
                                 - 0.0509999263371985 z_k1_s1027
                                 - 0.127999407125675 z_k1_s1030
                                 - 0.0989998088701068 z_k1_s1031
                                 - 0.165000007449529 z_k1_s1032
                                 - 0.136999923203714 z_k1_s1033
                                 - 0.169999533494398 z_k1_s1034
                                 - 0.147999467301258 z_k1_s1035
                                 - 0.073999831450434 z_k1_s1036
                                 - 0.0569998781009942 z_k1_s1037
                                 - 0.0540000349239387 z_k1_s1040
                                 - 0.0390000252228446 z_k1_s1041
                                 - 0.0510000329837199 z_k1_s1042
                                 - 0.0259999966152369 z_k1_s1043
                                 - 0.0420000271630635 z_k1_s1044
                                 - 0.0519999326304952 z_k1_s1045
                                 - 0.0150000097010941 z_k1_s1046
                                 - 0.0239999852217612 z_k1_s1047
                                 - 0.0579998391538733 z_k1_s1050
                                 - 0.0629999339084037 z_k1_s1051
                                 - 0.0779997981724352 z_k1_s1052
                                 - 0.0399999600370796 z_k1_s1053
                                 - 0.0579998391538733 z_k1_s1054
                                 - 0.04600003174256 z_k1_s1055
                                 - 0.0369999451843173 z_k1_s1056
                                 - 0.0399999600370796 z_k1_s1057
                                 - 0.110999715212687 z_k1_s1060
                                 - 0.0379999939286241 z_k1_s1061
                                 - 0.0869999426260811 z_k1_s1062
                                 - 0.066000200612773 z_k1_s1063
                                 - 0.0349999633553264 z_k1_s1064
                                 - 0.0319999799820063 z_k1_s1065
                                 - 0.0609998349907599 z_k1_s1066
                                 - 0.0420000032263693 z_k1_s1067
                                 - 0.0830000289105862 z_k1_s1070
                                 - 0.0329999746994145 z_k1_s1071
                                 - 0.0720000373441146 z_k1_s1072
                                 - 0.0860000728832289 z_k1_s1073
                                 - 0.0619997211020867 z_k1_s1074
                                 - 0.0769999409653006 z_k1_s1075
                                 - 0.053999773508258 z_k1_s1076
                                 - 0.0419999539174461 z_k1_s1077
                                 - 0.0779997700814215 z_k1_s1100
                                 - 0.066999810069939 z_k1_s1101
                                 - 0.0600000500626322 z_k1_s1102
                                 - 0.0320000070334038 z_k1_s1103
                                 - 0.0799998700835093 z_k1_s1104
                                 - 0.135999720141966 z_k1_s1105
                                 - 0.109000140113782 z_k1_s1106
                                 - 0.0279999840292283 z_k1_s1107
                                 - 0.0969996720167086 z_k1_s1110
                                 - 0.0779997010444842 z_k1_s1111
                                 - 0.191999282878666 z_k1_s1112
                                 - 0.193998856035087 z_k1_s1113
                                 - 0.290999504048456 z_k1_s1114
                                 - 0.116999551566726 z_k1_s1115
                                 - 0.163999158709643 z_k1_s1116
                                 - 0.0869997322416998 z_k1_s1117
                                 - 0.203999889888371 z_k1_s1120
                                 - 0.182000029135685 z_k1_s1121
                                 - 0.34400026922346 z_k1_s1122
                                 - 0.162999927940216 z_k1_s1123
                                 - 0.274998839408493 z_k1_s1124
                                 - 0.535998800883305 z_k1_s1125
                                 - 0.311000594094414 z_k1_s1126
                                 - 0.305001074979979 z_k1_s1127
                                 - 0.0900004550740517 z_k1_s1130
                                 - 0.297001077032406 z_k1_s1131
                                 - 0.085000436942193 z_k1_s1132
                                 - 0.191000692637002 z_k1_s1133
                                 - 0.166000601977708 z_k1_s1134
                                 - 0.203000736153463 z_k1_s1135
                                 - 0.163000591098593 z_k1_s1136
                                 - 0.0920004623267952 z_k1_s1137
                                 - 0.2430008980626 z_k1_s1140
                                 - 0.117000489511851 z_k1_s1141
                                 - 0.0890003558892908 z_k1_s1142
                                 - 0.200000770874695 z_k1_s1143
                                 - 0.0870003738806468 z_k1_s1144
                                 - 0.14700073364357 z_k1_s1145
                                 - 0.0690002788018213 z_k1_s1146
                                 - 0.333001373456728 z_k1_s1147
                                 - 0.223999851297456 z_k1_s1150
                                 - 0.0499999774056772 z_k1_s1151
                                 - 0.119000254845366 z_k1_s1152
                                 - 0.119999708373737 z_k1_s1153
                                 - 0.0929999817145485 z_k1_s1154
                                 - 0.224999304825828 z_k1_s1155
                                 - 0.237999322691293 z_k1_s1156
                                 - 0.117999970417387 z_k1_s1157
                                 - 0.0759997870881988 z_k1_s1160
                                 - 0.101999759249937 z_k1_s1161
                                 - 0.0770001367866215 z_k1_s1162
                                 - 0.0969997207573086 z_k1_s1163
                                 - 0.0679998394999583 z_k1_s1164
                                 - 0.0629998010073295 z_k1_s1165
                                 - 0.0359999921470135 z_k1_s1166
                                 - 0.0509999936249344 z_k1_s1167
                                 - 0.411000480380902 z_k1_s1170
                                 - 0.281000354739767 z_k1_s1171
                                 - 0.352999408878483 z_k1_s1172
                                 - 0.464000037011037 z_k1_s1173
                                 - 0.223000136738179 z_k1_s1174
                                 - 0.449001182399728 z_k1_s1175
                                 - 0.301999889196707 z_k1_s1176
                                 - 0.362000997647486 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.799785 w_k1_ADM_s000
                                 + 0.766052 w_k1_AXP_s000
                                 + 0.660833 w_k1_DIS_s000
                                 + 0.771621 w_k1_FDX_s000
                                 + 0.85023 w_k1_IBM_s000
                                 + 0.911886 w_k1_INTC_s000
                                 + 0.817239 w_k1_KO_s000
                                 + 0.702242 w_k1_MAT_s000
                                 + 0.787811 w_k1_MSFT_s000
                                 + 0.892184 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.0157 w_k1_ADM_s000
                                 + 0.957888 w_k1_AXP_s000
                                 + 0.941651 w_k1_DIS_s000
                                 + 1.08972 w_k1_FDX_s000
                                 + 0.883939 w_k1_IBM_s000
                                 + 1.0183 w_k1_INTC_s000
                                 + 0.982485 w_k1_KO_s000
                                 + 0.812051 w_k1_MAT_s000
                                 + 0.963455 w_k1_MSFT_s000
                                 + 1.00084 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.930617 w_k1_ADM_s000
                                 + 1.06903 w_k1_AXP_s000
                                 + 0.742854 w_k1_DIS_s000
                                 + 0.802175 w_k1_FDX_s000
                                 + 1.20799 w_k1_IBM_s000
                                 + 0.957291 w_k1_INTC_s000
                                 + 0.90675 w_k1_KO_s000 + 1.24497 w_k1_MAT_s000
                                 + 0.780294 w_k1_MSFT_s000
                                 + 0.934992 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.831621 w_k1_ADM_s000
                                 + 0.967396 w_k1_AXP_s000
                                 + 1.10245 w_k1_DIS_s000
                                 + 0.979858 w_k1_FDX_s000
                                 + 1.05246 w_k1_IBM_s000
                                 + 1.0388 w_k1_INTC_s000 + 1.01735 w_k1_KO_s000
                                 + 1.09608 w_k1_MAT_s000
                                 + 0.912759 w_k1_MSFT_s000
                                 + 1.00772 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.810753 w_k1_ADM_s000
                                 + 0.744338 w_k1_AXP_s000
                                 + 0.947767 w_k1_DIS_s000
                                 + 0.871089 w_k1_FDX_s000
                                 + 0.963256 w_k1_IBM_s000
                                 + 1.02687 w_k1_INTC_s000
                                 + 1.06462 w_k1_KO_s000
                                 + 0.775471 w_k1_MAT_s000
                                 + 0.919128 w_k1_MSFT_s000
                                 + 0.863066 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.14565 w_k1_ADM_s000
                                 + 1.28778 w_k1_AXP_s000
                                 + 1.44434 w_k1_DIS_s000 + 1.138 w_k1_FDX_s000
                                 + 1.47322 w_k1_IBM_s000
                                 + 0.996007 w_k1_INTC_s000
                                 + 1.28502 w_k1_KO_s000 + 1.08592 w_k1_MAT_s000
                                 + 1.06489 w_k1_MSFT_s000
                                 + 1.10502 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.872197 w_k1_ADM_s000
                                 + 0.534144 w_k1_AXP_s000
                                 + 0.768111 w_k1_DIS_s000
                                 + 0.801756 w_k1_FDX_s000
                                 + 0.648227 w_k1_IBM_s000
                                 + 0.835342 w_k1_INTC_s000
                                 + 0.746156 w_k1_KO_s000
                                 + 0.467418 w_k1_MAT_s000
                                 + 0.852082 w_k1_MSFT_s000
                                 + 0.823452 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.22354 w_k1_ADM_s000
                                 + 1.02616 w_k1_AXP_s000
                                 + 1.36485 w_k1_DIS_s000
                                 + 1.17601 w_k1_FDX_s000
                                 + 0.969466 w_k1_IBM_s000
                                 + 1.2606 w_k1_INTC_s000 + 1.09375 w_k1_KO_s000
                                 + 0.626404 w_k1_MAT_s000
                                 + 1.39202 w_k1_MSFT_s000
                                 + 1.10839 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.754107 w_k1_ADM_s001
                                 + 0.685441 w_k1_AXP_s001
                                 + 0.612784 w_k1_DIS_s001
                                 + 0.724652 w_k1_FDX_s001
                                 + 0.825379 w_k1_IBM_s001
                                 + 0.915473 w_k1_INTC_s001
                                 + 0.772171 w_k1_KO_s001
                                 + 0.773734 w_k1_MAT_s001
                                 + 0.848312 w_k1_MSFT_s001
                                 + 0.825155 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.01678 w_k1_ADM_s001
                                 + 1.14286 w_k1_AXP_s001 + 1.0128 w_k1_DIS_s001
                                 + 0.936439 w_k1_FDX_s001
                                 + 1.44665 w_k1_IBM_s001
                                 + 1.11445 w_k1_INTC_s001
                                 + 1.06834 w_k1_KO_s001 + 1.35598 w_k1_MAT_s001
                                 + 0.913414 w_k1_MSFT_s001
                                 + 1.03659 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.01895 w_k1_ADM_s001
                                 + 0.859776 w_k1_AXP_s001
                                 + 0.920374 w_k1_DIS_s001
                                 + 1.10018 w_k1_FDX_s001
                                 + 0.68171 w_k1_IBM_s001
                                 + 1.13265 w_k1_INTC_s001
                                 + 0.882021 w_k1_KO_s001
                                 + 0.517275 w_k1_MAT_s001
                                 + 1.12452 w_k1_MSFT_s001
                                 + 0.983545 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.896283 w_k1_ADM_s001
                                 + 0.910571 w_k1_AXP_s001
                                 + 0.752665 w_k1_DIS_s001
                                 + 0.872729 w_k1_FDX_s001
                                 + 0.796921 w_k1_IBM_s001
                                 + 0.879099 w_k1_INTC_s001
                                 + 0.756606 w_k1_KO_s001
                                 + 0.922251 w_k1_MAT_s001
                                 + 0.835877 w_k1_MSFT_s001
                                 + 0.879969 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.946744 w_k1_ADM_s001
                                 + 1.2079 w_k1_AXP_s001
                                 + 0.787449 w_k1_DIS_s001
                                 + 0.96397 w_k1_FDX_s001
                                 + 0.828313 w_k1_IBM_s001
                                 + 0.978825 w_k1_INTC_s001
                                 + 0.740552 w_k1_KO_s001
                                 + 1.10154 w_k1_MAT_s001
                                 + 0.859621 w_k1_MSFT_s001
                                 + 0.833945 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.21792 w_k1_ADM_s001
                                 + 1.37107 w_k1_AXP_s001
                                 + 1.37884 w_k1_DIS_s001
                                 + 1.28349 w_k1_FDX_s001
                                 + 1.18497 w_k1_IBM_s001
                                 + 1.11833 w_k1_INTC_s001
                                 + 1.07963 w_k1_KO_s001 + 1.09603 w_k1_MAT_s001
                                 + 1.23092 w_k1_MSFT_s001
                                 + 1.25353 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.824068 w_k1_ADM_s001
                                 + 0.91268 w_k1_AXP_s001
                                 + 1.04417 w_k1_DIS_s001
                                 + 0.874108 w_k1_FDX_s001
                                 + 1.22106 w_k1_IBM_s001
                                 + 0.855674 w_k1_INTC_s001
                                 + 0.940252 w_k1_KO_s001
                                 + 1.06535 w_k1_MAT_s001
                                 + 0.904515 w_k1_MSFT_s001
                                 + 0.905417 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.752378 w_k1_ADM_s001
                                 + 0.620747 w_k1_AXP_s001
                                 + 0.735056 w_k1_DIS_s001
                                 + 0.866548 w_k1_FDX_s001
                                 + 0.576458 w_k1_IBM_s001
                                 + 0.89403 w_k1_INTC_s001
                                 + 0.599671 w_k1_KO_s001
                                 + 0.495443 w_k1_MAT_s001
                                 + 0.851596 w_k1_MSFT_s001
                                 + 0.825055 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.945942 w_k1_ADM_s002
                                 + 0.916246 w_k1_AXP_s002
                                 + 1.24418 w_k1_DIS_s002
                                 + 1.06281 w_k1_FDX_s002
                                 + 1.13229 w_k1_IBM_s002
                                 + 1.03624 w_k1_INTC_s002
                                 + 1.01633 w_k1_KO_s002
                                 + 0.990188 w_k1_MAT_s002
                                 + 1.14442 w_k1_MSFT_s002
                                 + 1.02702 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.03911 w_k1_ADM_s002
                                 + 0.970214 w_k1_AXP_s002
                                 + 0.774163 w_k1_DIS_s002
                                 + 0.992168 w_k1_FDX_s002
                                 + 0.904029 w_k1_IBM_s002
                                 + 1.00191 w_k1_INTC_s002
                                 + 0.771712 w_k1_KO_s002
                                 + 0.858211 w_k1_MAT_s002
                                 + 0.936812 w_k1_MSFT_s002
                                 + 0.957608 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.04525 w_k1_ADM_s002
                                 + 1.15654 w_k1_AXP_s002
                                 + 1.00787 w_k1_DIS_s002
                                 + 1.02174 w_k1_FDX_s002
                                 + 1.23719 w_k1_IBM_s002
                                 + 0.978139 w_k1_INTC_s002
                                 + 0.875089 w_k1_KO_s002 + 1.3072 w_k1_MAT_s002
                                 + 0.959248 w_k1_MSFT_s002
                                 + 0.992283 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.709524 w_k1_ADM_s002
                                 + 0.535916 w_k1_AXP_s002
                                 + 0.599586 w_k1_DIS_s002
                                 + 0.708392 w_k1_FDX_s002
                                 + 0.652563 w_k1_IBM_s002
                                 + 0.858017 w_k1_INTC_s002
                                 + 0.633613 w_k1_KO_s002
                                 + 0.547945 w_k1_MAT_s002
                                 + 0.794254 w_k1_MSFT_s002
                                 + 0.801848 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.747314 w_k1_ADM_s002
                                 + 0.694759 w_k1_AXP_s002
                                 + 0.885243 w_k1_DIS_s002
                                 + 0.898869 w_k1_FDX_s002
                                 + 0.710579 w_k1_IBM_s002
                                 + 0.928889 w_k1_INTC_s002
                                 + 0.685787 w_k1_KO_s002
                                 + 0.672487 w_k1_MAT_s002
                                 + 0.965064 w_k1_MSFT_s002
                                 + 0.886919 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.11027 w_k1_ADM_s002
                                 + 1.35176 w_k1_AXP_s002
                                 + 1.44912 w_k1_DIS_s002
                                 + 1.34536 w_k1_FDX_s002
                                 + 1.22233 w_k1_IBM_s002
                                 + 1.42815 w_k1_INTC_s002
                                 + 1.18313 w_k1_KO_s002 + 1.11568 w_k1_MAT_s002
                                 + 1.38887 w_k1_MSFT_s002
                                 + 1.17189 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.638246 w_k1_ADM_s002
                                 + 0.752226 w_k1_AXP_s002
                                 + 0.615464 w_k1_DIS_s002
                                 + 0.707373 w_k1_FDX_s002
                                 + 0.88508 w_k1_IBM_s002
                                 + 0.948542 w_k1_INTC_s002
                                 + 0.713269 w_k1_KO_s002
                                 + 0.687412 w_k1_MAT_s002
                                 + 0.728762 w_k1_MSFT_s002
                                 + 0.830978 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.806412 w_k1_ADM_s002
                                 + 0.83695 w_k1_AXP_s002
                                 + 0.892243 w_k1_DIS_s002
                                 + 0.896359 w_k1_FDX_s002
                                 + 1.03381 w_k1_IBM_s002
                                 + 1.15312 w_k1_INTC_s002
                                 + 0.847917 w_k1_KO_s002
                                 + 1.10048 w_k1_MAT_s002
                                 + 0.927746 w_k1_MSFT_s002
                                 + 1.05855 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.39208 w_k1_ADM_s003
                                 + 1.47398 w_k1_AXP_s003
                                 + 1.53867 w_k1_DIS_s003
                                 + 1.26944 w_k1_FDX_s003
                                 + 1.35172 w_k1_IBM_s003
                                 + 1.14068 w_k1_INTC_s003
                                 + 1.39316 w_k1_KO_s003 + 1.46894 w_k1_MAT_s003
                                 + 1.29765 w_k1_MSFT_s003
                                 + 1.45131 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.1355 w_k1_ADM_s003
                                 + 1.16249 w_k1_AXP_s003
                                 + 0.966686 w_k1_DIS_s003
                                 + 1.06868 w_k1_FDX_s003
                                 + 0.838957 w_k1_IBM_s003
                                 + 1.1459 w_k1_INTC_s003
                                 + 0.861863 w_k1_KO_s003
                                 + 0.807938 w_k1_MAT_s003
                                 + 0.992745 w_k1_MSFT_s003
                                 + 1.06493 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.58608 w_k1_ADM_s003
                                 + 1.61555 w_k1_AXP_s003
                                 + 1.18478 w_k1_DIS_s003
                                 + 1.19029 w_k1_FDX_s003
                                 + 1.16553 w_k1_IBM_s003
                                 + 1.32313 w_k1_INTC_s003
                                 + 1.10118 w_k1_KO_s003 + 1.51446 w_k1_MAT_s003
                                 + 1.11289 w_k1_MSFT_s003
                                 + 1.36591 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.880357 w_k1_ADM_s003
                                 + 0.983207 w_k1_AXP_s003
                                 + 0.766301 w_k1_DIS_s003
                                 + 0.808339 w_k1_FDX_s003
                                 + 1.30192 w_k1_IBM_s003
                                 + 0.902817 w_k1_INTC_s003
                                 + 0.955019 w_k1_KO_s003
                                 + 1.39608 w_k1_MAT_s003
                                 + 0.838805 w_k1_MSFT_s003
                                 + 0.976806 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01329 w_k1_ADM_s003
                                 + 0.766388 w_k1_AXP_s003
                                 + 0.803226 w_k1_DIS_s003
                                 + 0.924693 w_k1_FDX_s003
                                 + 0.726997 w_k1_IBM_s003
                                 + 1.06822 w_k1_INTC_s003
                                 + 0.917077 w_k1_KO_s003
                                 + 0.614836 w_k1_MAT_s003
                                 + 1.00981 w_k1_MSFT_s003
                                 + 1.00947 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.944645 w_k1_ADM_s003
                                 + 0.621601 w_k1_AXP_s003
                                 + 0.671243 w_k1_DIS_s003
                                 + 0.760461 w_k1_FDX_s003
                                 + 0.850547 w_k1_IBM_s003
                                 + 0.761496 w_k1_INTC_s003
                                 + 0.79593 w_k1_KO_s003
                                 + 0.895682 w_k1_MAT_s003
                                 + 0.751534 w_k1_MSFT_s003
                                 + 0.88073 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.967144 w_k1_ADM_s003
                                 + 0.971788 w_k1_AXP_s003
                                 + 0.945822 w_k1_DIS_s003
                                 + 0.912573 w_k1_FDX_s003
                                 + 1.15945 w_k1_IBM_s003
                                 + 0.932386 w_k1_INTC_s003
                                 + 1.16706 w_k1_KO_s003 + 1.22635 w_k1_MAT_s003
                                 + 0.997251 w_k1_MSFT_s003
                                 + 1.07891 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.1694 w_k1_ADM_s003
                                 + 1.03207 w_k1_AXP_s003
                                 + 1.28544 w_k1_DIS_s003
                                 + 1.09022 w_k1_FDX_s003
                                 + 1.14723 w_k1_IBM_s003
                                 + 1.23765 w_k1_INTC_s003
                                 + 1.24629 w_k1_KO_s003 + 1.34147 w_k1_MAT_s003
                                 + 1.12227 w_k1_MSFT_s003
                                 + 1.23784 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.07542 w_k1_ADM_s004
                                 + 1.21201 w_k1_AXP_s004
                                 + 1.22913 w_k1_DIS_s004
                                 + 1.18923 w_k1_FDX_s004
                                 + 0.859728 w_k1_IBM_s004
                                 + 1.10956 w_k1_INTC_s004
                                 + 1.10434 w_k1_KO_s004
                                 + 0.739722 w_k1_MAT_s004
                                 + 0.998172 w_k1_MSFT_s004
                                 + 1.0515 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.679331 w_k1_ADM_s004
                                 + 0.623414 w_k1_AXP_s004
                                 + 0.717762 w_k1_DIS_s004
                                 + 0.693122 w_k1_FDX_s004
                                 + 0.851899 w_k1_IBM_s004
                                 + 0.832289 w_k1_INTC_s004
                                 + 0.869729 w_k1_KO_s004
                                 + 0.525674 w_k1_MAT_s004
                                 + 0.828475 w_k1_MSFT_s004
                                 + 0.813174 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.936523 w_k1_ADM_s004
                                 + 1.50943 w_k1_AXP_s004 + 1.5016 w_k1_DIS_s004
                                 + 1.20942 w_k1_FDX_s004
                                 + 1.35167 w_k1_IBM_s004
                                 + 1.28211 w_k1_INTC_s004
                                 + 1.54748 w_k1_KO_s004 + 1.42078 w_k1_MAT_s004
                                 + 1.17453 w_k1_MSFT_s004
                                 + 1.15111 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.779608 w_k1_ADM_s004
                                 + 0.600192 w_k1_AXP_s004
                                 + 0.702288 w_k1_DIS_s004
                                 + 0.817637 w_k1_FDX_s004
                                 + 0.621078 w_k1_IBM_s004
                                 + 0.914019 w_k1_INTC_s004
                                 + 0.724017 w_k1_KO_s004
                                 + 0.371847 w_k1_MAT_s004
                                 + 0.822936 w_k1_MSFT_s004
                                 + 0.785483 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.959386 w_k1_ADM_s004
                                 + 0.981429 w_k1_AXP_s004
                                 + 0.809043 w_k1_DIS_s004
                                 + 0.918343 w_k1_FDX_s004
                                 + 0.860582 w_k1_IBM_s004
                                 + 0.776766 w_k1_INTC_s004
                                 + 0.967001 w_k1_KO_s004 + 1.0641 w_k1_MAT_s004
                                 + 0.851372 w_k1_MSFT_s004
                                 + 0.905906 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.929388 w_k1_ADM_s004
                                 + 0.832769 w_k1_AXP_s004
                                 + 1.4596 w_k1_DIS_s004 + 1.10083 w_k1_FDX_s004
                                 + 0.966539 w_k1_IBM_s004
                                 + 1.10826 w_k1_INTC_s004
                                 + 1.27353 w_k1_KO_s004 + 0.78607 w_k1_MAT_s004
                                 + 1.28513 w_k1_MSFT_s004
                                 + 1.12429 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.947746 w_k1_ADM_s004
                                 + 1.05088 w_k1_AXP_s004
                                 + 0.995746 w_k1_DIS_s004
                                 + 0.920038 w_k1_FDX_s004
                                 + 1.34899 w_k1_IBM_s004
                                 + 1.0711 w_k1_INTC_s004 + 1.14083 w_k1_KO_s004
                                 + 1.19596 w_k1_MAT_s004
                                 + 0.98976 w_k1_MSFT_s004
                                 + 1.01459 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.863092 w_k1_ADM_s004
                                 + 0.862947 w_k1_AXP_s004
                                 + 1.04129 w_k1_DIS_s004
                                 + 0.970066 w_k1_FDX_s004
                                 + 0.937615 w_k1_IBM_s004
                                 + 1.04111 w_k1_INTC_s004
                                 + 1.00305 w_k1_KO_s004
                                 + 0.972628 w_k1_MAT_s004
                                 + 1.01808 w_k1_MSFT_s004
                                 + 0.977593 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.847554 w_k1_ADM_s005
                                 + 0.925547 w_k1_AXP_s005
                                 + 0.745888 w_k1_DIS_s005
                                 + 0.767807 w_k1_FDX_s005
                                 + 1.40064 w_k1_IBM_s005
                                 + 0.81647 w_k1_INTC_s005
                                 + 0.924958 w_k1_KO_s005
                                 + 1.32467 w_k1_MAT_s005
                                 + 0.76878 w_k1_MSFT_s005
                                 + 0.927045 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.990467 w_k1_ADM_s005
                                 + 1.11879 w_k1_AXP_s005
                                 + 0.787339 w_k1_DIS_s005
                                 + 0.948181 w_k1_FDX_s005
                                 + 1.01009 w_k1_IBM_s005
                                 + 1.03267 w_k1_INTC_s005
                                 + 0.925349 w_k1_KO_s005 + 1.3551 w_k1_MAT_s005
                                 + 1.00466 w_k1_MSFT_s005
                                 + 0.962535 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.11856 w_k1_ADM_s005
                                 + 1.18785 w_k1_AXP_s005
                                 + 1.03195 w_k1_DIS_s005
                                 + 1.00769 w_k1_FDX_s005
                                 + 1.70142 w_k1_IBM_s005
                                 + 1.29747 w_k1_INTC_s005
                                 + 1.30838 w_k1_KO_s005 + 2.2447 w_k1_MAT_s005
                                 + 1.14318 w_k1_MSFT_s005
                                 + 1.20169 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.845222 w_k1_ADM_s005
                                 + 0.75017 w_k1_AXP_s005
                                 + 0.738894 w_k1_DIS_s005
                                 + 0.836375 w_k1_FDX_s005
                                 + 0.667533 w_k1_IBM_s005
                                 + 0.859885 w_k1_INTC_s005
                                 + 0.684525 w_k1_KO_s005
                                 + 0.754207 w_k1_MAT_s005
                                 + 0.862039 w_k1_MSFT_s005
                                 + 0.862941 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.02661 w_k1_ADM_s005
                                 + 0.916311 w_k1_AXP_s005
                                 + 1.03078 w_k1_DIS_s005
                                 + 0.89552 w_k1_FDX_s005
                                 + 1.17736 w_k1_IBM_s005
                                 + 1.00724 w_k1_INTC_s005
                                 + 1.02583 w_k1_KO_s005 + 0.92134 w_k1_MAT_s005
                                 + 0.981964 w_k1_MSFT_s005
                                 + 0.981515 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.22675 w_k1_ADM_s005
                                 + 1.45687 w_k1_AXP_s005 + 1.7632 w_k1_DIS_s005
                                 + 1.43566 w_k1_FDX_s005
                                 + 1.13688 w_k1_IBM_s005
                                 + 1.27383 w_k1_INTC_s005 + 1.1925 w_k1_KO_s005
                                 + 1.40364 w_k1_MAT_s005
                                 + 1.36368 w_k1_MSFT_s005
                                 + 1.18905 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.27137 w_k1_ADM_s005
                                 + 1.36221 w_k1_AXP_s005
                                 + 1.24671 w_k1_DIS_s005
                                 + 1.24167 w_k1_FDX_s005
                                 + 1.21252 w_k1_IBM_s005
                                 + 1.29402 w_k1_INTC_s005
                                 + 1.18542 w_k1_KO_s005 + 1.84057 w_k1_MAT_s005
                                 + 1.27907 w_k1_MSFT_s005
                                 + 1.34686 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.0936 w_k1_ADM_s005
                                 + 0.88735 w_k1_AXP_s005
                                 + 1.21181 w_k1_DIS_s005
                                 + 1.07137 w_k1_FDX_s005
                                 + 0.885402 w_k1_IBM_s005
                                 + 1.27652 w_k1_INTC_s005
                                 + 1.04114 w_k1_KO_s005
                                 + 0.798462 w_k1_MAT_s005
                                 + 1.127 w_k1_MSFT_s005 + 0.991931 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.813024 w_k1_ADM_s006
                                 + 0.872359 w_k1_AXP_s006
                                 + 0.993612 w_k1_DIS_s006
                                 + 0.870176 w_k1_FDX_s006
                                 + 1.32973 w_k1_IBM_s006
                                 + 1.16399 w_k1_INTC_s006
                                 + 1.29303 w_k1_KO_s006 + 1.07382 w_k1_MAT_s006
                                 + 1.00237 w_k1_MSFT_s006
                                 + 0.981605 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.954457 w_k1_ADM_s006
                                 + 1.03382 w_k1_AXP_s006
                                 + 0.684625 w_k1_DIS_s006
                                 + 0.858926 w_k1_FDX_s006
                                 + 0.917138 w_k1_IBM_s006
                                 + 1.13023 w_k1_INTC_s006
                                 + 0.912311 w_k1_KO_s006
                                 + 0.885127 w_k1_MAT_s006
                                 + 0.870718 w_k1_MSFT_s006
                                 + 0.832923 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.24618 w_k1_ADM_s006
                                 + 1.17835 w_k1_AXP_s006
                                 + 1.34017 w_k1_DIS_s006
                                 + 1.36234 w_k1_FDX_s006
                                 + 0.843676 w_k1_IBM_s006
                                 + 1.29326 w_k1_INTC_s006
                                 + 1.09909 w_k1_KO_s006
                                 + 0.887361 w_k1_MAT_s006
                                 + 1.19767 w_k1_MSFT_s006
                                 + 0.949063 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.13535 w_k1_ADM_s006
                                 + 0.949074 w_k1_AXP_s006
                                 + 0.915303 w_k1_DIS_s006
                                 + 1.03109 w_k1_FDX_s006
                                 + 0.853673 w_k1_IBM_s006
                                 + 1.2099 w_k1_INTC_s006 + 1.03659 w_k1_KO_s006
                                 + 0.611902 w_k1_MAT_s006
                                 + 1.03532 w_k1_MSFT_s006
                                 + 0.914577 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.36124 w_k1_ADM_s006
                                 + 1.72411 w_k1_AXP_s006
                                 + 1.74624 w_k1_DIS_s006
                                 + 1.38921 w_k1_FDX_s006
                                 + 1.47017 w_k1_IBM_s006
                                 + 1.52532 w_k1_INTC_s006
                                 + 1.44965 w_k1_KO_s006 + 1.42749 w_k1_MAT_s006
                                 + 1.20353 w_k1_MSFT_s006
                                 + 1.25979 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.893207 w_k1_ADM_s006
                                 + 0.697456 w_k1_AXP_s006
                                 + 0.607824 w_k1_DIS_s006
                                 + 0.807107 w_k1_FDX_s006
                                 + 0.587583 w_k1_IBM_s006
                                 + 0.863276 w_k1_INTC_s006
                                 + 0.723983 w_k1_KO_s006
                                 + 0.503049 w_k1_MAT_s006
                                 + 0.806749 w_k1_MSFT_s006
                                 + 0.737875 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.999827 w_k1_ADM_s006
                                 + 1.20946 w_k1_AXP_s006
                                 + 1.06791 w_k1_DIS_s006 + 1.0505 w_k1_FDX_s006
                                 + 1.47516 w_k1_IBM_s006
                                 + 1.14786 w_k1_INTC_s006
                                 + 1.18124 w_k1_KO_s006 + 1.34567 w_k1_MAT_s006
                                 + 0.8532 w_k1_MSFT_s006 + 0.99421 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.03952 w_k1_ADM_s006
                                 + 1.2346 w_k1_AXP_s006 + 1.35409 w_k1_DIS_s006
                                 + 1.10644 w_k1_FDX_s006
                                 + 1.35568 w_k1_IBM_s006
                                 + 1.44944 w_k1_INTC_s006
                                 + 1.41901 w_k1_KO_s006 + 1.09687 w_k1_MAT_s006
                                 + 1.13492 w_k1_MSFT_s006
                                 + 1.06161 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.715984 w_k1_ADM_s007
                                 + 0.703894 w_k1_AXP_s007
                                 + 0.833502 w_k1_DIS_s007
                                 + 0.78387 w_k1_FDX_s007
                                 + 1.37355 w_k1_IBM_s007
                                 + 0.698487 w_k1_INTC_s007
                                 + 0.981805 w_k1_KO_s007
                                 + 1.07854 w_k1_MAT_s007
                                 + 0.890056 w_k1_MSFT_s007
                                 + 0.982794 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.759621 w_k1_ADM_s007
                                 + 0.807028 w_k1_AXP_s007
                                 + 0.955962 w_k1_DIS_s007
                                 + 0.90803 w_k1_FDX_s007
                                 + 0.848226 w_k1_IBM_s007
                                 + 1.06204 w_k1_INTC_s007
                                 + 1.03913 w_k1_KO_s007
                                 + 0.687053 w_k1_MAT_s007
                                 + 1.08256 w_k1_MSFT_s007
                                 + 0.989456 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.2809 w_k1_ADM_s007
                                 + 1.58032 w_k1_AXP_s007 + 2.0007 w_k1_DIS_s007
                                 + 1.52452 w_k1_FDX_s007 + 1.4546 w_k1_IBM_s007
                                 + 1.05784 w_k1_INTC_s007
                                 + 1.37875 w_k1_KO_s007 + 1.64057 w_k1_MAT_s007
                                 + 1.45108 w_k1_MSFT_s007
                                 + 1.18311 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.610437 w_k1_ADM_s007
                                 + 0.608932 w_k1_AXP_s007
                                 + 0.529925 w_k1_DIS_s007
                                 + 0.686088 w_k1_FDX_s007
                                 + 0.739464 w_k1_IBM_s007
                                 + 0.613914 w_k1_INTC_s007
                                 + 0.674959 w_k1_KO_s007
                                 + 0.782599 w_k1_MAT_s007
                                 + 0.73661 w_k1_MSFT_s007
                                 + 0.743546 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.733284 w_k1_ADM_s007
                                 + 0.705422 w_k1_AXP_s007
                                 + 0.649818 w_k1_DIS_s007
                                 + 0.826067 w_k1_FDX_s007
                                 + 0.807528 w_k1_IBM_s007
                                 + 0.745169 w_k1_INTC_s007
                                 + 0.755473 w_k1_KO_s007
                                 + 0.710987 w_k1_MAT_s007
                                 + 0.898627 w_k1_MSFT_s007
                                 + 0.892811 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.757596 w_k1_ADM_s007
                                 + 1.05176 w_k1_AXP_s007
                                 + 0.898742 w_k1_DIS_s007
                                 + 0.963847 w_k1_FDX_s007
                                 + 1.11081 w_k1_IBM_s007
                                 + 1.01821 w_k1_INTC_s007
                                 + 0.918826 w_k1_KO_s007
                                 + 0.996959 w_k1_MAT_s007
                                 + 0.945611 w_k1_MSFT_s007
                                 + 1.01315 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.936662 w_k1_ADM_s007
                                 + 1.10422 w_k1_AXP_s007
                                 + 1.14679 w_k1_DIS_s007
                                 + 1.05466 w_k1_FDX_s007
                                 + 1.60659 w_k1_IBM_s007
                                 + 1.13359 w_k1_INTC_s007 + 1.1114 w_k1_KO_s007
                                 + 1.18015 w_k1_MAT_s007
                                 + 1.11003 w_k1_MSFT_s007
                                 + 1.06933 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.957387 w_k1_ADM_s007
                                 + 0.960636 w_k1_AXP_s007
                                 + 1.06672 w_k1_DIS_s007 + 1.1745 w_k1_FDX_s007
                                 + 0.823984 w_k1_IBM_s007
                                 + 0.976887 w_k1_INTC_s007
                                 + 0.898179 w_k1_KO_s007
                                 + 0.714486 w_k1_MAT_s007
                                 + 1.09768 w_k1_MSFT_s007 + 1.0974 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.038 w_k1_ADM_s010
                                 + 0.812527 w_k1_AXP_s010
                                 + 0.818817 w_k1_DIS_s010
                                 + 0.757285 w_k1_FDX_s010
                                 + 1.01706 w_k1_IBM_s010
                                 + 0.963175 w_k1_INTC_s010
                                 + 1.0702 w_k1_KO_s010 + 0.947975 w_k1_MAT_s010
                                 + 0.959642 w_k1_MSFT_s010
                                 + 0.918157 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.10375 w_k1_ADM_s010
                                 + 1.0717 w_k1_AXP_s010 + 1.24017 w_k1_DIS_s010
                                 + 1.1176 w_k1_FDX_s010
                                 + 0.933462 w_k1_IBM_s010
                                 + 1.02946 w_k1_INTC_s010
                                 + 1.09641 w_k1_KO_s010
                                 + 0.837455 w_k1_MAT_s010
                                 + 1.13565 w_k1_MSFT_s010
                                 + 1.14109 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.834314 w_k1_ADM_s010
                                 + 0.644244 w_k1_AXP_s010
                                 + 0.79132 w_k1_DIS_s010
                                 + 0.864713 w_k1_FDX_s010
                                 + 0.726616 w_k1_IBM_s010
                                 + 0.95438 w_k1_INTC_s010
                                 + 0.783627 w_k1_KO_s010
                                 + 0.595984 w_k1_MAT_s010
                                 + 1.12714 w_k1_MSFT_s010
                                 + 0.977018 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.18812 w_k1_ADM_s010
                                 + 1.24953 w_k1_AXP_s010
                                 + 1.56589 w_k1_DIS_s010
                                 + 1.21379 w_k1_FDX_s010
                                 + 1.23275 w_k1_IBM_s010
                                 + 1.3944 w_k1_INTC_s010 + 1.35418 w_k1_KO_s010
                                 + 1.34935 w_k1_MAT_s010
                                 + 1.35687 w_k1_MSFT_s010
                                 + 1.21737 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.21379 w_k1_ADM_s010
                                 + 1.12428 w_k1_AXP_s010
                                 + 1.25976 w_k1_DIS_s010 + 1.067 w_k1_FDX_s010
                                 + 1.29999 w_k1_IBM_s010
                                 + 1.16988 w_k1_INTC_s010
                                 + 1.25371 w_k1_KO_s010 + 1.21992 w_k1_MAT_s010
                                 + 1.23268 w_k1_MSFT_s010
                                 + 1.24345 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.878854 w_k1_ADM_s010
                                 + 1.20847 w_k1_AXP_s010
                                 + 1.01923 w_k1_DIS_s010
                                 + 1.03311 w_k1_FDX_s010
                                 + 1.09784 w_k1_IBM_s010
                                 + 1.09136 w_k1_INTC_s010
                                 + 1.08801 w_k1_KO_s010 + 1.3416 w_k1_MAT_s010
                                 + 1.01771 w_k1_MSFT_s010
                                 + 1.14054 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.06451 w_k1_ADM_s010
                                 + 1.23035 w_k1_AXP_s010
                                 + 1.16507 w_k1_DIS_s010
                                 + 0.975654 w_k1_FDX_s010
                                 + 1.59735 w_k1_IBM_s010
                                 + 1.10963 w_k1_INTC_s010
                                 + 1.26165 w_k1_KO_s010 + 1.37163 w_k1_MAT_s010
                                 + 0.973315 w_k1_MSFT_s010
                                 + 0.964146 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.01276 w_k1_ADM_s010
                                 + 0.952603 w_k1_AXP_s010
                                 + 1.24842 w_k1_DIS_s010
                                 + 1.16914 w_k1_FDX_s010
                                 + 0.692904 w_k1_IBM_s010
                                 + 0.884788 w_k1_INTC_s010
                                 + 0.919728 w_k1_KO_s010
                                 + 0.830177 w_k1_MAT_s010
                                 + 1.18278 w_k1_MSFT_s010
                                 + 1.04692 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.985972 w_k1_ADM_s011
                                 + 0.89513 w_k1_AXP_s011
                                 + 1.30886 w_k1_DIS_s011
                                 + 1.08558 w_k1_FDX_s011
                                 + 0.967182 w_k1_IBM_s011
                                 + 0.929539 w_k1_INTC_s011
                                 + 0.908959 w_k1_KO_s011
                                 + 0.981324 w_k1_MAT_s011
                                 + 1.12282 w_k1_MSFT_s011 + 1.0352 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.881903 w_k1_ADM_s011
                                 + 0.919161 w_k1_AXP_s011
                                 + 0.92456 w_k1_DIS_s011
                                 + 0.84092 w_k1_FDX_s011
                                 + 1.33121 w_k1_IBM_s011
                                 + 0.778928 w_k1_INTC_s011
                                 + 1.01915 w_k1_KO_s011 + 1.47411 w_k1_MAT_s011
                                 + 0.86404 w_k1_MSFT_s011
                                 + 0.960719 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.25318 w_k1_ADM_s011
                                 + 1.0317 w_k1_AXP_s011 + 1.12063 w_k1_DIS_s011
                                 + 1.26535 w_k1_FDX_s011
                                 + 0.81123 w_k1_IBM_s011
                                 + 1.23107 w_k1_INTC_s011
                                 + 0.873206 w_k1_KO_s011
                                 + 0.951309 w_k1_MAT_s011
                                 + 1.15328 w_k1_MSFT_s011
                                 + 0.978712 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.02249 w_k1_ADM_s011
                                 + 0.963084 w_k1_AXP_s011
                                 + 1.00286 w_k1_DIS_s011
                                 + 1.02391 w_k1_FDX_s011
                                 + 1.16507 w_k1_IBM_s011
                                 + 0.858534 w_k1_INTC_s011
                                 + 0.988374 w_k1_KO_s011
                                 + 1.65912 w_k1_MAT_s011
                                 + 1.07417 w_k1_MSFT_s011
                                 + 1.05823 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.872153 w_k1_ADM_s011
                                 + 0.704546 w_k1_AXP_s011
                                 + 0.686657 w_k1_DIS_s011
                                 + 0.847706 w_k1_FDX_s011
                                 + 0.809234 w_k1_IBM_s011
                                 + 0.831037 w_k1_INTC_s011
                                 + 0.732907 w_k1_KO_s011
                                 + 0.654857 w_k1_MAT_s011
                                 + 0.864084 w_k1_MSFT_s011
                                 + 0.839144 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.13931 w_k1_ADM_s011
                                 + 1.61726 w_k1_AXP_s011
                                 + 1.17434 w_k1_DIS_s011
                                 + 1.24827 w_k1_FDX_s011
                                 + 1.44187 w_k1_IBM_s011
                                 + 1.1858 w_k1_INTC_s011 + 1.16537 w_k1_KO_s011
                                 + 2.10238 w_k1_MAT_s011
                                 + 1.19694 w_k1_MSFT_s011
                                 + 1.07104 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.27247 w_k1_ADM_s011
                                 + 1.16405 w_k1_AXP_s011
                                 + 1.31657 w_k1_DIS_s011
                                 + 1.30676 w_k1_FDX_s011
                                 + 1.09804 w_k1_IBM_s011
                                 + 1.29239 w_k1_INTC_s011
                                 + 1.09848 w_k1_KO_s011 + 1.26248 w_k1_MAT_s011
                                 + 1.47016 w_k1_MSFT_s011 + 1.1524 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.12829 w_k1_ADM_s011
                                 + 0.728121 w_k1_AXP_s011
                                 + 0.954869 w_k1_DIS_s011
                                 + 0.859023 w_k1_FDX_s011
                                 + 0.836468 w_k1_IBM_s011
                                 + 0.817516 w_k1_INTC_s011
                                 + 0.84861 w_k1_KO_s011
                                 + 0.848756 w_k1_MAT_s011
                                 + 0.978153 w_k1_MSFT_s011
                                 + 0.963888 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.986256 w_k1_ADM_s012
                                 + 0.654095 w_k1_AXP_s012
                                 + 0.715444 w_k1_DIS_s012
                                 + 0.847599 w_k1_FDX_s012
                                 + 0.812482 w_k1_IBM_s012
                                 + 0.915281 w_k1_INTC_s012
                                 + 0.851843 w_k1_KO_s012
                                 + 0.742886 w_k1_MAT_s012
                                 + 0.896531 w_k1_MSFT_s012
                                 + 0.890606 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.891691 w_k1_ADM_s012
                                 + 0.831682 w_k1_AXP_s012
                                 + 0.744036 w_k1_DIS_s012
                                 + 0.822324 w_k1_FDX_s012
                                 + 1.21532 w_k1_IBM_s012
                                 + 0.899366 w_k1_INTC_s012
                                 + 0.954768 w_k1_KO_s012
                                 + 1.09679 w_k1_MAT_s012
                                 + 0.798809 w_k1_MSFT_s012
                                 + 0.955489 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.25693 w_k1_ADM_s012
                                 + 1.51069 w_k1_AXP_s012
                                 + 1.31404 w_k1_DIS_s012
                                 + 1.34242 w_k1_FDX_s012
                                 + 1.26408 w_k1_IBM_s012
                                 + 1.24045 w_k1_INTC_s012
                                 + 1.16615 w_k1_KO_s012 + 1.4871 w_k1_MAT_s012
                                 + 1.14971 w_k1_MSFT_s012 + 1.2709 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.994487 w_k1_ADM_s012
                                 + 0.965187 w_k1_AXP_s012
                                 + 1.06525 w_k1_DIS_s012
                                 + 1.00316 w_k1_FDX_s012
                                 + 1.58778 w_k1_IBM_s012
                                 + 0.848548 w_k1_INTC_s012
                                 + 1.16118 w_k1_KO_s012 + 1.49952 w_k1_MAT_s012
                                 + 0.905174 w_k1_MSFT_s012
                                 + 1.08534 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.16434 w_k1_ADM_s012
                                 + 0.838569 w_k1_AXP_s012
                                 + 0.885841 w_k1_DIS_s012
                                 + 1.13688 w_k1_FDX_s012
                                 + 0.692703 w_k1_IBM_s012
                                 + 1.01017 w_k1_INTC_s012
                                 + 0.712117 w_k1_KO_s012
                                 + 0.782264 w_k1_MAT_s012
                                 + 1.10348 w_k1_MSFT_s012
                                 + 0.995249 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.21278 w_k1_ADM_s012
                                 + 1.07308 w_k1_AXP_s012
                                 + 1.21433 w_k1_DIS_s012
                                 + 1.35894 w_k1_FDX_s012
                                 + 0.896463 w_k1_IBM_s012
                                 + 1.16304 w_k1_INTC_s012
                                 + 1.08962 w_k1_KO_s012
                                 + 0.918799 w_k1_MAT_s012
                                 + 1.3146 w_k1_MSFT_s012 + 1.06246 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.14909 w_k1_ADM_s012
                                 + 0.994474 w_k1_AXP_s012
                                 + 1.85842 w_k1_DIS_s012
                                 + 1.34659 w_k1_FDX_s012 + 1.196 w_k1_IBM_s012
                                 + 1.34815 w_k1_INTC_s012
                                 + 1.39029 w_k1_KO_s012 + 1.03494 w_k1_MAT_s012
                                 + 1.3921 w_k1_MSFT_s012 + 1.17911 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.991784 w_k1_ADM_s012
                                 + 0.938173 w_k1_AXP_s012
                                 + 0.972213 w_k1_DIS_s012
                                 + 0.96676 w_k1_FDX_s012
                                 + 1.03995 w_k1_IBM_s012
                                 + 1.03448 w_k1_INTC_s012
                                 + 0.879195 w_k1_KO_s012
                                 + 0.910233 w_k1_MAT_s012
                                 + 0.94471 w_k1_MSFT_s012
                                 + 0.974694 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.721726 w_k1_ADM_s013
                                 + 1.49718 w_k1_AXP_s013
                                 + 1.67777 w_k1_DIS_s013
                                 + 1.36612 w_k1_FDX_s013
                                 + 1.82486 w_k1_IBM_s013
                                 + 1.4173 w_k1_INTC_s013 + 1.53375 w_k1_KO_s013
                                 + 1.85087 w_k1_MAT_s013
                                 + 1.31642 w_k1_MSFT_s013
                                 + 1.12181 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.13487 w_k1_ADM_s013
                                 + 1.25849 w_k1_AXP_s013
                                 + 1.04687 w_k1_DIS_s013
                                 + 1.34286 w_k1_FDX_s013
                                 + 0.734354 w_k1_IBM_s013
                                 + 1.39176 w_k1_INTC_s013
                                 + 0.963371 w_k1_KO_s013
                                 + 0.624205 w_k1_MAT_s013
                                 + 1.15695 w_k1_MSFT_s013
                                 + 0.988104 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.06088 w_k1_ADM_s013
                                 + 0.749002 w_k1_AXP_s013
                                 + 0.748127 w_k1_DIS_s013
                                 + 0.971684 w_k1_FDX_s013
                                 + 0.604934 w_k1_IBM_s013
                                 + 1.04904 w_k1_INTC_s013
                                 + 0.834417 w_k1_KO_s013
                                 + 0.574279 w_k1_MAT_s013
                                 + 0.899679 w_k1_MSFT_s013
                                 + 0.867885 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17826 w_k1_ADM_s013
                                 + 1.32916 w_k1_AXP_s013
                                 + 1.22954 w_k1_DIS_s013
                                 + 1.30749 w_k1_FDX_s013
                                 + 1.28739 w_k1_IBM_s013
                                 + 1.54709 w_k1_INTC_s013
                                 + 1.32474 w_k1_KO_s013 + 1.25092 w_k1_MAT_s013
                                 + 1.2563 w_k1_MSFT_s013 + 1.20078 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.08883 w_k1_ADM_s013
                                 + 1.30121 w_k1_AXP_s013
                                 + 1.27079 w_k1_DIS_s013
                                 + 1.17573 w_k1_FDX_s013
                                 + 1.09012 w_k1_IBM_s013
                                 + 1.18175 w_k1_INTC_s013
                                 + 1.15741 w_k1_KO_s013 + 1.11742 w_k1_MAT_s013
                                 + 1.17491 w_k1_MSFT_s013
                                 + 1.04699 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.03825 w_k1_ADM_s013
                                 + 0.851814 w_k1_AXP_s013
                                 + 0.878473 w_k1_DIS_s013
                                 + 0.866175 w_k1_FDX_s013
                                 + 1.3452 w_k1_IBM_s013
                                 + 0.876741 w_k1_INTC_s013
                                 + 0.966078 w_k1_KO_s013 + 1.0216 w_k1_MAT_s013
                                 + 0.964298 w_k1_MSFT_s013
                                 + 0.960877 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.921825 w_k1_ADM_s013
                                 + 0.693347 w_k1_AXP_s013
                                 + 0.683126 w_k1_DIS_s013
                                 + 0.782317 w_k1_FDX_s013
                                 + 0.810817 w_k1_IBM_s013
                                 + 0.864072 w_k1_INTC_s013
                                 + 0.762021 w_k1_KO_s013
                                 + 0.588454 w_k1_MAT_s013
                                 + 0.803062 w_k1_MSFT_s013
                                 + 0.853753 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.975791 w_k1_ADM_s013
                                 + 0.881669 w_k1_AXP_s013
                                 + 0.960703 w_k1_DIS_s013
                                 + 1.00273 w_k1_FDX_s013
                                 + 0.915392 w_k1_IBM_s013
                                 + 1.0772 w_k1_INTC_s013
                                 + 0.984086 w_k1_KO_s013
                                 + 0.752229 w_k1_MAT_s013
                                 + 1.0273 w_k1_MSFT_s013
                                 + 0.893828 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.893337 w_k1_ADM_s014
                                 + 0.848515 w_k1_AXP_s014
                                 + 0.793893 w_k1_DIS_s014
                                 + 0.850728 w_k1_FDX_s014
                                 + 0.773431 w_k1_IBM_s014
                                 + 0.88087 w_k1_INTC_s014
                                 + 1.00056 w_k1_KO_s014
                                 + 0.749112 w_k1_MAT_s014
                                 + 0.953022 w_k1_MSFT_s014
                                 + 0.852634 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.09753 w_k1_ADM_s014
                                 + 1.17967 w_k1_AXP_s014
                                 + 1.35699 w_k1_DIS_s014 + 1.1838 w_k1_FDX_s014
                                 + 1.23729 w_k1_IBM_s014
                                 + 1.14488 w_k1_INTC_s014 + 1.2145 w_k1_KO_s014
                                 + 1.11315 w_k1_MAT_s014
                                 + 1.18066 w_k1_MSFT_s014
                                 + 0.971871 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.828494 w_k1_ADM_s014
                                 + 0.519044 w_k1_AXP_s014
                                 + 0.593461 w_k1_DIS_s014
                                 + 0.654153 w_k1_FDX_s014
                                 + 0.505155 w_k1_IBM_s014
                                 + 0.73215 w_k1_INTC_s014
                                 + 0.656526 w_k1_KO_s014
                                 + 0.326572 w_k1_MAT_s014
                                 + 0.803631 w_k1_MSFT_s014
                                 + 0.696328 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.905203 w_k1_ADM_s014
                                 + 0.686923 w_k1_AXP_s014
                                 + 0.669071 w_k1_DIS_s014
                                 + 0.81031 w_k1_FDX_s014
                                 + 0.991339 w_k1_IBM_s014
                                 + 0.840366 w_k1_INTC_s014
                                 + 0.948219 w_k1_KO_s014
                                 + 0.817664 w_k1_MAT_s014
                                 + 0.846573 w_k1_MSFT_s014
                                 + 0.832781 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.21591 w_k1_ADM_s014
                                 + 1.19226 w_k1_AXP_s014
                                 + 1.44019 w_k1_DIS_s014 + 1.2276 w_k1_FDX_s014
                                 + 0.893154 w_k1_IBM_s014
                                 + 1.29701 w_k1_INTC_s014
                                 + 1.26284 w_k1_KO_s014
                                 + 0.940088 w_k1_MAT_s014
                                 + 1.34954 w_k1_MSFT_s014
                                 + 1.06659 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.40915 w_k1_ADM_s014
                                 + 1.49962 w_k1_AXP_s014
                                 + 1.79042 w_k1_DIS_s014
                                 + 1.37938 w_k1_FDX_s014
                                 + 1.64816 w_k1_IBM_s014
                                 + 1.42516 w_k1_INTC_s014
                                 + 1.52412 w_k1_KO_s014 + 1.34987 w_k1_MAT_s014
                                 + 1.476 w_k1_MSFT_s014 + 1.19454 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.857859 w_k1_ADM_s014
                                 + 0.852631 w_k1_AXP_s014
                                 + 0.906585 w_k1_DIS_s014
                                 + 0.850057 w_k1_FDX_s014
                                 + 1.48391 w_k1_IBM_s014
                                 + 1.04464 w_k1_INTC_s014
                                 + 1.23447 w_k1_KO_s014 + 1.33755 w_k1_MAT_s014
                                 + 0.891767 w_k1_MSFT_s014
                                 + 0.985831 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.10959 w_k1_ADM_s014
                                 + 0.973448 w_k1_AXP_s014
                                 + 0.947377 w_k1_DIS_s014
                                 + 1.03562 w_k1_FDX_s014
                                 + 0.690392 w_k1_IBM_s014
                                 + 1.06282 w_k1_INTC_s014
                                 + 1.00192 w_k1_KO_s014
                                 + 0.714616 w_k1_MAT_s014
                                 + 1.16329 w_k1_MSFT_s014
                                 + 0.959197 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.82655 w_k1_ADM_s015
                                 + 0.647081 w_k1_AXP_s015
                                 + 0.808723 w_k1_DIS_s015
                                 + 0.777813 w_k1_FDX_s015
                                 + 0.915058 w_k1_IBM_s015
                                 + 0.91693 w_k1_INTC_s015 + 1.0015 w_k1_KO_s015
                                 + 0.852418 w_k1_MAT_s015
                                 + 1.07268 w_k1_MSFT_s015
                                 + 1.00617 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.02752 w_k1_ADM_s015
                                 + 0.693801 w_k1_AXP_s015
                                 + 1.02899 w_k1_DIS_s015
                                 + 0.936737 w_k1_FDX_s015
                                 + 0.802506 w_k1_IBM_s015
                                 + 0.806468 w_k1_INTC_s015
                                 + 0.927601 w_k1_KO_s015
                                 + 0.545744 w_k1_MAT_s015
                                 + 1.12605 w_k1_MSFT_s015
                                 + 0.98654 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.04862 w_k1_ADM_s015
                                 + 1.32412 w_k1_AXP_s015
                                 + 1.36252 w_k1_DIS_s015
                                 + 1.25389 w_k1_FDX_s015 + 1.3615 w_k1_IBM_s015
                                 + 0.909297 w_k1_INTC_s015
                                 + 1.24509 w_k1_KO_s015 + 1.35708 w_k1_MAT_s015
                                 + 1.19926 w_k1_MSFT_s015
                                 + 1.24547 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.944792 w_k1_ADM_s015
                                 + 0.946461 w_k1_AXP_s015
                                 + 1.13808 w_k1_DIS_s015
                                 + 1.13283 w_k1_FDX_s015
                                 + 0.78114 w_k1_IBM_s015
                                 + 1.10256 w_k1_INTC_s015
                                 + 1.04254 w_k1_KO_s015 + 0.66931 w_k1_MAT_s015
                                 + 1.20919 w_k1_MSFT_s015
                                 + 1.14895 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.909142 w_k1_ADM_s015
                                 + 0.991895 w_k1_AXP_s015
                                 + 1.12559 w_k1_DIS_s015
                                 + 0.995319 w_k1_FDX_s015
                                 + 1.21627 w_k1_IBM_s015
                                 + 1.06599 w_k1_INTC_s015
                                 + 1.28613 w_k1_KO_s015 + 1.22771 w_k1_MAT_s015
                                 + 1.1879 w_k1_MSFT_s015 + 1.18962 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.973691 w_k1_ADM_s015
                                 + 1.17721 w_k1_AXP_s015
                                 + 1.13167 w_k1_DIS_s015
                                 + 0.975796 w_k1_FDX_s015
                                 + 1.5526 w_k1_IBM_s015
                                 + 0.935057 w_k1_INTC_s015
                                 + 1.45919 w_k1_KO_s015 + 1.90594 w_k1_MAT_s015
                                 + 1.20541 w_k1_MSFT_s015
                                 + 1.28094 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.7655 w_k1_ADM_s015
                                 + 1.0585 w_k1_AXP_s015
                                 + 0.708724 w_k1_DIS_s015
                                 + 0.843613 w_k1_FDX_s015
                                 + 1.09427 w_k1_IBM_s015
                                 + 0.958989 w_k1_INTC_s015
                                 + 0.986891 w_k1_KO_s015
                                 + 1.25473 w_k1_MAT_s015
                                 + 1.01871 w_k1_MSFT_s015
                                 + 1.03016 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.63435 w_k1_ADM_s015
                                 + 1.37999 w_k1_AXP_s015
                                 + 1.43447 w_k1_DIS_s015
                                 + 1.40356 w_k1_FDX_s015
                                 + 1.04286 w_k1_IBM_s015
                                 + 1.26902 w_k1_INTC_s015
                                 + 1.28994 w_k1_KO_s015 + 1.09029 w_k1_MAT_s015
                                 + 1.34174 w_k1_MSFT_s015
                                 + 1.24487 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.846702 w_k1_ADM_s016
                                 + 0.876397 w_k1_AXP_s016
                                 + 1.11467 w_k1_DIS_s016 + 1.0303 w_k1_FDX_s016
                                 + 1.07916 w_k1_IBM_s016
                                 + 1.09908 w_k1_INTC_s016
                                 + 1.16734 w_k1_KO_s016 + 1.0302 w_k1_MAT_s016
                                 + 1.08936 w_k1_MSFT_s016
                                 + 0.937092 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.38661 w_k1_ADM_s016
                                 + 1.33505 w_k1_AXP_s016
                                 + 1.42097 w_k1_DIS_s016
                                 + 1.32745 w_k1_FDX_s016
                                 + 1.12967 w_k1_IBM_s016
                                 + 1.18855 w_k1_INTC_s016
                                 + 1.22203 w_k1_KO_s016 + 1.01469 w_k1_MAT_s016
                                 + 1.17315 w_k1_MSFT_s016
                                 + 1.14157 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.13256 w_k1_ADM_s016
                                 + 1.2203 w_k1_AXP_s016 + 1.05513 w_k1_DIS_s016
                                 + 1.1423 w_k1_FDX_s016
                                 + 0.888687 w_k1_IBM_s016
                                 + 0.992032 w_k1_INTC_s016
                                 + 0.966619 w_k1_KO_s016
                                 + 1.08883 w_k1_MAT_s016
                                 + 0.9915 w_k1_MSFT_s016 + 1.00442 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.06423 w_k1_ADM_s016
                                 + 0.952337 w_k1_AXP_s016
                                 + 0.811862 w_k1_DIS_s016
                                 + 1.01592 w_k1_FDX_s016
                                 + 0.833502 w_k1_IBM_s016
                                 + 1.05692 w_k1_INTC_s016
                                 + 0.883679 w_k1_KO_s016
                                 + 0.935248 w_k1_MAT_s016
                                 + 0.898303 w_k1_MSFT_s016
                                 + 1.01761 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.08009 w_k1_ADM_s016
                                 + 1.21903 w_k1_AXP_s016
                                 + 1.37953 w_k1_DIS_s016
                                 + 1.15066 w_k1_FDX_s016
                                 + 1.62801 w_k1_IBM_s016
                                 + 1.13033 w_k1_INTC_s016
                                 + 1.44808 w_k1_KO_s016 + 1.04408 w_k1_MAT_s016
                                 + 1.11714 w_k1_MSFT_s016
                                 + 1.09541 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.766082 w_k1_ADM_s016
                                 + 0.732464 w_k1_AXP_s016
                                 + 0.575248 w_k1_DIS_s016
                                 + 0.725789 w_k1_FDX_s016
                                 + 1.04417 w_k1_IBM_s016
                                 + 0.669384 w_k1_INTC_s016
                                 + 0.77368 w_k1_KO_s016
                                 + 0.687736 w_k1_MAT_s016
                                 + 0.609583 w_k1_MSFT_s016
                                 + 0.742283 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.770196 w_k1_ADM_s016
                                 + 0.780886 w_k1_AXP_s016
                                 + 0.77574 w_k1_DIS_s016
                                 + 0.838852 w_k1_FDX_s016
                                 + 0.768595 w_k1_IBM_s016
                                 + 0.864808 w_k1_INTC_s016
                                 + 0.721352 w_k1_KO_s016
                                 + 0.630994 w_k1_MAT_s016
                                 + 0.805555 w_k1_MSFT_s016
                                 + 0.850523 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.903572 w_k1_ADM_s016
                                 + 0.890343 w_k1_AXP_s016
                                 + 0.753122 w_k1_DIS_s016
                                 + 0.880278 w_k1_FDX_s016
                                 + 1.14282 w_k1_IBM_s016
                                 + 0.917992 w_k1_INTC_s016
                                 + 0.963674 w_k1_KO_s016
                                 + 1.02771 w_k1_MAT_s016
                                 + 0.892866 w_k1_MSFT_s016
                                 + 0.986173 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.977582 w_k1_ADM_s017
                                 + 0.761391 w_k1_AXP_s017
                                 + 0.85327 w_k1_DIS_s017
                                 + 1.02496 w_k1_FDX_s017
                                 + 0.626916 w_k1_IBM_s017
                                 + 0.906124 w_k1_INTC_s017
                                 + 0.80248 w_k1_KO_s017
                                 + 0.786341 w_k1_MAT_s017
                                 + 0.950659 w_k1_MSFT_s017
                                 + 0.911456 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.805881 w_k1_ADM_s017
                                 + 0.594618 w_k1_AXP_s017
                                 + 0.569204 w_k1_DIS_s017
                                 + 0.826073 w_k1_FDX_s017
                                 + 0.616076 w_k1_IBM_s017
                                 + 0.728484 w_k1_INTC_s017
                                 + 0.595044 w_k1_KO_s017
                                 + 0.687137 w_k1_MAT_s017
                                 + 0.803521 w_k1_MSFT_s017
                                 + 0.867603 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.927391 w_k1_ADM_s017
                                 + 1.13815 w_k1_AXP_s017
                                 + 1.12042 w_k1_DIS_s017
                                 + 1.15231 w_k1_FDX_s017
                                 + 1.09953 w_k1_IBM_s017
                                 + 0.98828 w_k1_INTC_s017
                                 + 1.13804 w_k1_KO_s017 + 1.65776 w_k1_MAT_s017
                                 + 1.06924 w_k1_MSFT_s017
                                 + 1.12825 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.03928 w_k1_ADM_s017
                                 + 1.09953 w_k1_AXP_s017
                                 + 1.06285 w_k1_DIS_s017
                                 + 1.02395 w_k1_FDX_s017
                                 + 1.41815 w_k1_IBM_s017
                                 + 1.03364 w_k1_INTC_s017
                                 + 1.21393 w_k1_KO_s017 + 1.26772 w_k1_MAT_s017
                                 + 1.04582 w_k1_MSFT_s017
                                 + 1.02102 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.697941 w_k1_ADM_s017
                                 + 0.769419 w_k1_AXP_s017
                                 + 0.558233 w_k1_DIS_s017
                                 + 0.712473 w_k1_FDX_s017
                                 + 0.856141 w_k1_IBM_s017
                                 + 0.67948 w_k1_INTC_s017
                                 + 0.650223 w_k1_KO_s017
                                 + 1.00467 w_k1_MAT_s017
                                 + 0.638118 w_k1_MSFT_s017
                                 + 0.814442 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.13535 w_k1_ADM_s017
                                 + 0.898047 w_k1_AXP_s017
                                 + 1.10841 w_k1_DIS_s017
                                 + 1.08247 w_k1_FDX_s017
                                 + 0.886863 w_k1_IBM_s017
                                 + 1.00355 w_k1_INTC_s017
                                 + 0.889221 w_k1_KO_s017
                                 + 0.778712 w_k1_MAT_s017
                                 + 0.995026 w_k1_MSFT_s017
                                 + 1.04812 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.783554 w_k1_ADM_s017
                                 + 0.795032 w_k1_AXP_s017
                                 + 0.728059 w_k1_DIS_s017
                                 + 0.780924 w_k1_FDX_s017
                                 + 1.20199 w_k1_IBM_s017
                                 + 0.729843 w_k1_INTC_s017
                                 + 0.905716 w_k1_KO_s017
                                 + 1.48843 w_k1_MAT_s017
                                 + 0.796711 w_k1_MSFT_s017
                                 + 0.843114 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.15305 w_k1_ADM_s017
                                 + 1.36896 w_k1_AXP_s017
                                 + 1.67619 w_k1_DIS_s017 + 1.3269 w_k1_FDX_s017
                                 + 1.55153 w_k1_IBM_s017
                                 + 1.33512 w_k1_INTC_s017 + 1.5596 w_k1_KO_s017
                                 + 2.25256 w_k1_MAT_s017
                                 + 1.30285 w_k1_MSFT_s017
                                 + 1.40692 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.5737 w_k1_ADM_s100
                                 + 0.989273 w_k1_AXP_s100
                                 + 1.45875 w_k1_DIS_s100
                                 + 1.39261 w_k1_FDX_s100 + 1.1154 w_k1_IBM_s100
                                 + 0.874505 w_k1_INTC_s100
                                 + 1.36191 w_k1_KO_s100 + 1.26224 w_k1_MAT_s100
                                 + 1.24403 w_k1_MSFT_s100
                                 + 1.16342 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.04117 w_k1_ADM_s100
                                 + 0.982646 w_k1_AXP_s100
                                 + 0.916807 w_k1_DIS_s100
                                 + 0.976144 w_k1_FDX_s100
                                 + 1.09017 w_k1_IBM_s100
                                 + 0.932241 w_k1_INTC_s100
                                 + 1.13761 w_k1_KO_s100 + 1.39865 w_k1_MAT_s100
                                 + 0.979719 w_k1_MSFT_s100
                                 + 0.991275 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.724619 w_k1_ADM_s100
                                 + 0.664843 w_k1_AXP_s100
                                 + 0.592284 w_k1_DIS_s100
                                 + 0.670208 w_k1_FDX_s100
                                 + 0.876788 w_k1_IBM_s100
                                 + 0.734829 w_k1_INTC_s100
                                 + 0.802245 w_k1_KO_s100
                                 + 0.843625 w_k1_MAT_s100
                                 + 0.788486 w_k1_MSFT_s100
                                 + 0.884363 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.0339 w_k1_ADM_s100
                                 + 0.897216 w_k1_AXP_s100
                                 + 0.864883 w_k1_DIS_s100
                                 + 0.816785 w_k1_FDX_s100
                                 + 1.44563 w_k1_IBM_s100
                                 + 0.76331 w_k1_INTC_s100
                                 + 0.988827 w_k1_KO_s100
                                 + 0.942075 w_k1_MAT_s100
                                 + 0.744983 w_k1_MSFT_s100
                                 + 0.958213 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09997 w_k1_ADM_s100
                                 + 0.78265 w_k1_AXP_s100
                                 + 0.996161 w_k1_DIS_s100
                                 + 1.03179 w_k1_FDX_s100
                                 + 0.624722 w_k1_IBM_s100
                                 + 0.808024 w_k1_INTC_s100
                                 + 0.870022 w_k1_KO_s100
                                 + 0.457242 w_k1_MAT_s100
                                 + 1.00066 w_k1_MSFT_s100
                                 + 1.02486 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.911098 w_k1_ADM_s100
                                 + 0.614346 w_k1_AXP_s100
                                 + 0.97006 w_k1_DIS_s100
                                 + 0.860876 w_k1_FDX_s100
                                 + 0.900616 w_k1_IBM_s100
                                 + 0.73454 w_k1_INTC_s100 + 1.0158 w_k1_KO_s100
                                 + 0.653262 w_k1_MAT_s100
                                 + 0.923028 w_k1_MSFT_s100
                                 + 0.959959 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.72521 w_k1_ADM_s100
                                 + 1.73703 w_k1_AXP_s100
                                 + 1.26989 w_k1_DIS_s100
                                 + 1.37924 w_k1_FDX_s100
                                 + 1.25513 w_k1_IBM_s100
                                 + 1.20738 w_k1_INTC_s100
                                 + 1.05951 w_k1_KO_s100 + 1.08715 w_k1_MAT_s100
                                 + 1.00509 w_k1_MSFT_s100
                                 + 1.17476 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.29617 w_k1_ADM_s100
                                 + 1.01199 w_k1_AXP_s100
                                 + 0.878178 w_k1_DIS_s100
                                 + 1.08979 w_k1_FDX_s100
                                 + 0.88372 w_k1_IBM_s100
                                 + 0.743618 w_k1_INTC_s100
                                 + 0.893684 w_k1_KO_s100
                                 + 1.14205 w_k1_MAT_s100
                                 + 0.868836 w_k1_MSFT_s100
                                 + 1.09791 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.46369 w_k1_ADM_s101
                                 + 1.29683 w_k1_AXP_s101
                                 + 1.35208 w_k1_DIS_s101
                                 + 1.20231 w_k1_FDX_s101
                                 + 1.27189 w_k1_IBM_s101
                                 + 1.21205 w_k1_INTC_s101
                                 + 1.29441 w_k1_KO_s101 + 1.16157 w_k1_MAT_s101
                                 + 0.957992 w_k1_MSFT_s101
                                 + 1.25756 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.01522 w_k1_ADM_s101
                                 + 0.992097 w_k1_AXP_s101
                                 + 1.11819 w_k1_DIS_s101
                                 + 1.15803 w_k1_FDX_s101
                                 + 0.707927 w_k1_IBM_s101
                                 + 1.20964 w_k1_INTC_s101
                                 + 0.979553 w_k1_KO_s101
                                 + 0.76145 w_k1_MAT_s101
                                 + 1.2114 w_k1_MSFT_s101 + 1.1679 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.935988 w_k1_ADM_s101
                                 + 0.64411 w_k1_AXP_s101
                                 + 0.653633 w_k1_DIS_s101
                                 + 0.775734 w_k1_FDX_s101
                                 + 0.943718 w_k1_IBM_s101
                                 + 0.847188 w_k1_INTC_s101
                                 + 0.859473 w_k1_KO_s101
                                 + 0.693092 w_k1_MAT_s101
                                 + 0.810442 w_k1_MSFT_s101
                                 + 0.88207 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.12436 w_k1_ADM_s101
                                 + 1.10578 w_k1_AXP_s101
                                 + 1.13146 w_k1_DIS_s101
                                 + 1.00909 w_k1_FDX_s101
                                 + 1.59784 w_k1_IBM_s101
                                 + 1.25104 w_k1_INTC_s101
                                 + 1.43151 w_k1_KO_s101 + 1.80635 w_k1_MAT_s101
                                 + 1.08685 w_k1_MSFT_s101
                                 + 1.13654 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.11729 w_k1_ADM_s101
                                 + 1.09954 w_k1_AXP_s101
                                 + 0.930098 w_k1_DIS_s101
                                 + 1.04641 w_k1_FDX_s101
                                 + 1.07619 w_k1_IBM_s101
                                 + 1.08331 w_k1_INTC_s101
                                 + 1.11964 w_k1_KO_s101 + 1.13817 w_k1_MAT_s101
                                 + 0.961175 w_k1_MSFT_s101
                                 + 1.12355 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.96193 w_k1_ADM_s101
                                 + 0.80432 w_k1_AXP_s101
                                 + 0.946984 w_k1_DIS_s101
                                 + 0.934658 w_k1_FDX_s101
                                 + 0.969839 w_k1_IBM_s101
                                 + 1.25203 w_k1_INTC_s101
                                 + 1.03646 w_k1_KO_s101
                                 + 0.722648 w_k1_MAT_s101
                                 + 0.971175 w_k1_MSFT_s101
                                 + 1.04225 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.882095 w_k1_ADM_s101
                                 + 0.677124 w_k1_AXP_s101
                                 + 0.529053 w_k1_DIS_s101
                                 + 0.718635 w_k1_FDX_s101
                                 + 0.717807 w_k1_IBM_s101
                                 + 0.702279 w_k1_INTC_s101
                                 + 0.863113 w_k1_KO_s101
                                 + 0.962283 w_k1_MAT_s101
                                 + 0.81476 w_k1_MSFT_s101
                                 + 0.92844 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.754512 w_k1_ADM_s101
                                 + 0.845964 w_k1_AXP_s101
                                 + 0.800311 w_k1_DIS_s101
                                 + 0.801509 w_k1_FDX_s101
                                 + 1.09529 w_k1_IBM_s101
                                 + 0.837462 w_k1_INTC_s101
                                 + 1.19957 w_k1_KO_s101 + 1.41338 w_k1_MAT_s101
                                 + 0.866415 w_k1_MSFT_s101
                                 + 1.15235 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02748 w_k1_ADM_s102
                                 + 1.33639 w_k1_AXP_s102
                                 + 1.15865 w_k1_DIS_s102
                                 + 1.22529 w_k1_FDX_s102
                                 + 1.40003 w_k1_IBM_s102
                                 + 1.51698 w_k1_INTC_s102
                                 + 1.24834 w_k1_KO_s102 + 1.19402 w_k1_MAT_s102
                                 + 1.19 w_k1_MSFT_s102 + 1.15644 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.794713 w_k1_ADM_s102
                                 + 0.772222 w_k1_AXP_s102
                                 + 0.659432 w_k1_DIS_s102
                                 + 0.834093 w_k1_FDX_s102
                                 + 0.667225 w_k1_IBM_s102
                                 + 0.98508 w_k1_INTC_s102
                                 + 0.63536 w_k1_KO_s102
                                 + 0.382009 w_k1_MAT_s102
                                 + 0.774794 w_k1_MSFT_s102
                                 + 0.740937 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.06357 w_k1_ADM_s102
                                 + 1.21041 w_k1_AXP_s102
                                 + 1.23155 w_k1_DIS_s102
                                 + 1.00673 w_k1_FDX_s102
                                 + 1.17245 w_k1_IBM_s102
                                 + 1.14599 w_k1_INTC_s102
                                 + 1.08109 w_k1_KO_s102 + 1.00113 w_k1_MAT_s102
                                 + 0.961579 w_k1_MSFT_s102
                                 + 0.826754 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.00974 w_k1_ADM_s102
                                 + 0.742351 w_k1_AXP_s102
                                 + 0.862554 w_k1_DIS_s102
                                 + 0.870509 w_k1_FDX_s102
                                 + 0.972212 w_k1_IBM_s102
                                 + 1.12486 w_k1_INTC_s102
                                 + 0.922759 w_k1_KO_s102
                                 + 0.616226 w_k1_MAT_s102
                                 + 0.820244 w_k1_MSFT_s102
                                 + 0.884461 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.961575 w_k1_ADM_s102
                                 + 0.998711 w_k1_AXP_s102
                                 + 1.63931 w_k1_DIS_s102
                                 + 1.24496 w_k1_FDX_s102
                                 + 1.10279 w_k1_IBM_s102
                                 + 1.57017 w_k1_INTC_s102
                                 + 1.33267 w_k1_KO_s102
                                 + 0.544851 w_k1_MAT_s102
                                 + 1.33077 w_k1_MSFT_s102
                                 + 1.05948 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.562438 w_k1_ADM_s102
                                 + 0.944155 w_k1_AXP_s102
                                 + 0.935823 w_k1_DIS_s102
                                 + 0.788502 w_k1_FDX_s102
                                 + 1.29847 w_k1_IBM_s102
                                 + 1.07101 w_k1_INTC_s102
                                 + 0.862377 w_k1_KO_s102
                                 + 0.831229 w_k1_MAT_s102
                                 + 0.790127 w_k1_MSFT_s102
                                 + 0.785725 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.12401 w_k1_ADM_s102
                                 + 1.14461 w_k1_AXP_s102
                                 + 1.22288 w_k1_DIS_s102
                                 + 1.02071 w_k1_FDX_s102
                                 + 1.77738 w_k1_IBM_s102
                                 + 0.924018 w_k1_INTC_s102
                                 + 1.2926 w_k1_KO_s102 + 1.3741 w_k1_MAT_s102
                                 + 0.933729 w_k1_MSFT_s102
                                 + 1.18918 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.861837 w_k1_ADM_s102
                                 + 0.893569 w_k1_AXP_s102
                                 + 1.10796 w_k1_DIS_s102 + 1.0879 w_k1_FDX_s102
                                 + 0.841077 w_k1_IBM_s102
                                 + 1.09364 w_k1_INTC_s102
                                 + 1.08132 w_k1_KO_s102 + 1.07883 w_k1_MAT_s102
                                 + 1.07948 w_k1_MSFT_s102
                                 + 0.954089 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.982081 w_k1_ADM_s103
                                 + 1.24612 w_k1_AXP_s103
                                 + 1.20231 w_k1_DIS_s103
                                 + 1.09992 w_k1_FDX_s103
                                 + 1.72905 w_k1_IBM_s103
                                 + 1.15656 w_k1_INTC_s103
                                 + 1.32099 w_k1_KO_s103 + 1.79651 w_k1_MAT_s103
                                 + 0.974196 w_k1_MSFT_s103
                                 + 1.2843 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.789966 w_k1_ADM_s103
                                 + 0.990047 w_k1_AXP_s103
                                 + 1.02339 w_k1_DIS_s103
                                 + 0.930284 w_k1_FDX_s103
                                 + 1.00365 w_k1_IBM_s103
                                 + 1.0869 w_k1_INTC_s103 + 1.1455 w_k1_KO_s103
                                 + 1.14575 w_k1_MAT_s103
                                 + 0.9609 w_k1_MSFT_s103 + 1.04154 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.964572 w_k1_ADM_s103
                                 + 1.03608 w_k1_AXP_s103
                                 + 1.19904 w_k1_DIS_s103
                                 + 1.19842 w_k1_FDX_s103
                                 + 0.864173 w_k1_IBM_s103
                                 + 1.35621 w_k1_INTC_s103
                                 + 0.885378 w_k1_KO_s103
                                 + 0.769919 w_k1_MAT_s103
                                 + 1.05757 w_k1_MSFT_s103
                                 + 1.03277 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.680805 w_k1_ADM_s103
                                 + 0.64944 w_k1_AXP_s103
                                 + 0.603069 w_k1_DIS_s103
                                 + 0.749791 w_k1_FDX_s103
                                 + 0.785697 w_k1_IBM_s103
                                 + 0.905139 w_k1_INTC_s103
                                 + 0.80593 w_k1_KO_s103
                                 + 0.862094 w_k1_MAT_s103
                                 + 0.932626 w_k1_MSFT_s103
                                 + 0.87997 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.753964 w_k1_ADM_s103
                                 + 0.903066 w_k1_AXP_s103
                                 + 0.801428 w_k1_DIS_s103
                                 + 0.979407 w_k1_FDX_s103
                                 + 0.666098 w_k1_IBM_s103
                                 + 1.13715 w_k1_INTC_s103
                                 + 0.89014 w_k1_KO_s103
                                 + 0.685832 w_k1_MAT_s103
                                 + 1.03815 w_k1_MSFT_s103
                                 + 0.867422 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.818228 w_k1_ADM_s103
                                 + 0.969901 w_k1_AXP_s103
                                 + 1.2883 w_k1_DIS_s103 + 1.09546 w_k1_FDX_s103
                                 + 1.25309 w_k1_IBM_s103
                                 + 0.980136 w_k1_INTC_s103
                                 + 1.27243 w_k1_KO_s103 + 1.4018 w_k1_MAT_s103
                                 + 1.07051 w_k1_MSFT_s103
                                 + 1.12112 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.731927 w_k1_ADM_s103
                                 + 0.608275 w_k1_AXP_s103
                                 + 0.865908 w_k1_DIS_s103
                                 + 0.844504 w_k1_FDX_s103
                                 + 1.12543 w_k1_IBM_s103
                                 + 0.838082 w_k1_INTC_s103
                                 + 0.94711 w_k1_KO_s103
                                 + 0.915004 w_k1_MAT_s103
                                 + 0.772233 w_k1_MSFT_s103
                                 + 0.985253 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.978714 w_k1_ADM_s103
                                 + 1.16555 w_k1_AXP_s103
                                 + 0.786444 w_k1_DIS_s103
                                 + 0.920941 w_k1_FDX_s103
                                 + 1.01893 w_k1_IBM_s103
                                 + 1.05234 w_k1_INTC_s103
                                 + 0.992011 w_k1_KO_s103
                                 + 1.24358 w_k1_MAT_s103
                                 + 0.933772 w_k1_MSFT_s103
                                 + 1.04861 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.06976 w_k1_ADM_s104
                                 + 0.989062 w_k1_AXP_s104
                                 + 0.939831 w_k1_DIS_s104 + 1.029 w_k1_FDX_s104
                                 + 1.03567 w_k1_IBM_s104
                                 + 0.812387 w_k1_INTC_s104
                                 + 1.06725 w_k1_KO_s104
                                 + 0.800636 w_k1_MAT_s104
                                 + 0.868054 w_k1_MSFT_s104
                                 + 1.00656 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.29588 w_k1_ADM_s104
                                 + 0.817065 w_k1_AXP_s104
                                 + 1.21312 w_k1_DIS_s104
                                 + 1.20621 w_k1_FDX_s104
                                 + 0.739528 w_k1_IBM_s104
                                 + 0.912813 w_k1_INTC_s104
                                 + 1.00837 w_k1_KO_s104
                                 + 0.467044 w_k1_MAT_s104
                                 + 0.969854 w_k1_MSFT_s104
                                 + 1.0073 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.50224 w_k1_ADM_s104
                                 + 1.35494 w_k1_AXP_s104 + 1.4667 w_k1_DIS_s104
                                 + 1.4691 w_k1_FDX_s104 + 1.30357 w_k1_IBM_s104
                                 + 1.29297 w_k1_INTC_s104 + 1.4604 w_k1_KO_s104
                                 + 0.992102 w_k1_MAT_s104
                                 + 1.22709 w_k1_MSFT_s104
                                 + 1.28253 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.35791 w_k1_ADM_s104
                                 + 1.50988 w_k1_AXP_s104
                                 + 0.878367 w_k1_DIS_s104
                                 + 0.998925 w_k1_FDX_s104
                                 + 1.12277 w_k1_IBM_s104
                                 + 0.758228 w_k1_INTC_s104
                                 + 0.965497 w_k1_KO_s104 + 1.2042 w_k1_MAT_s104
                                 + 0.817907 w_k1_MSFT_s104
                                 + 0.924975 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.923194 w_k1_ADM_s104
                                 + 0.736767 w_k1_AXP_s104
                                 + 0.7231 w_k1_DIS_s104 + 0.86947 w_k1_FDX_s104
                                 + 0.717755 w_k1_IBM_s104
                                 + 0.696004 w_k1_INTC_s104
                                 + 0.809167 w_k1_KO_s104
                                 + 0.567346 w_k1_MAT_s104
                                 + 0.674803 w_k1_MSFT_s104
                                 + 0.861049 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.03234 w_k1_ADM_s104
                                 + 0.998894 w_k1_AXP_s104
                                 + 1.02742 w_k1_DIS_s104
                                 + 0.958669 w_k1_FDX_s104
                                 + 1.31521 w_k1_IBM_s104
                                 + 0.860115 w_k1_INTC_s104
                                 + 1.17029 w_k1_KO_s104
                                 + 0.948665 w_k1_MAT_s104
                                 + 0.928212 w_k1_MSFT_s104
                                 + 0.98501 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.13471 w_k1_ADM_s104
                                 + 0.920729 w_k1_AXP_s104
                                 + 0.703105 w_k1_DIS_s104
                                 + 1.03709 w_k1_FDX_s104
                                 + 0.89045 w_k1_IBM_s104
                                 + 1.00976 w_k1_INTC_s104
                                 + 0.860257 w_k1_KO_s104
                                 + 0.721606 w_k1_MAT_s104
                                 + 0.87773 w_k1_MSFT_s104
                                 + 0.915249 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.938962 w_k1_ADM_s104
                                 + 0.751705 w_k1_AXP_s104
                                 + 0.57332 w_k1_DIS_s104
                                 + 0.805765 w_k1_FDX_s104
                                 + 1.01445 w_k1_IBM_s104
                                 + 0.755789 w_k1_INTC_s104
                                 + 0.731296 w_k1_KO_s104
                                 + 0.872062 w_k1_MAT_s104
                                 + 0.724226 w_k1_MSFT_s104
                                 + 0.826606 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.875437 w_k1_ADM_s105
                                 + 0.916725 w_k1_AXP_s105
                                 + 0.798979 w_k1_DIS_s105
                                 + 0.900572 w_k1_FDX_s105
                                 + 0.644832 w_k1_IBM_s105
                                 + 1.22586 w_k1_INTC_s105
                                 + 0.785281 w_k1_KO_s105
                                 + 0.748832 w_k1_MAT_s105
                                 + 0.860992 w_k1_MSFT_s105
                                 + 0.912277 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.10883 w_k1_ADM_s105
                                 + 1.2979 w_k1_AXP_s105
                                 + 0.806941 w_k1_DIS_s105
                                 + 0.969403 w_k1_FDX_s105
                                 + 1.03415 w_k1_IBM_s105
                                 + 1.31812 w_k1_INTC_s105
                                 + 0.853263 w_k1_KO_s105
                                 + 0.823919 w_k1_MAT_s105
                                 + 0.888889 w_k1_MSFT_s105
                                 + 1.06178 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.1099 w_k1_ADM_s105
                                 + 1.50968 w_k1_AXP_s105
                                 + 1.19517 w_k1_DIS_s105
                                 + 1.15789 w_k1_FDX_s105
                                 + 1.29927 w_k1_IBM_s105 + 1.201 w_k1_INTC_s105
                                 + 1.23423 w_k1_KO_s105 + 1.64598 w_k1_MAT_s105
                                 + 1.08499 w_k1_MSFT_s105
                                 + 1.25846 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.997303 w_k1_ADM_s105
                                 + 1.22587 w_k1_AXP_s105
                                 + 1.53413 w_k1_DIS_s105
                                 + 1.18653 w_k1_FDX_s105
                                 + 1.17695 w_k1_IBM_s105
                                 + 1.63428 w_k1_INTC_s105
                                 + 1.18153 w_k1_KO_s105
                                 + 0.943956 w_k1_MAT_s105
                                 + 1.18353 w_k1_MSFT_s105
                                 + 1.18495 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.840882 w_k1_ADM_s105
                                 + 1.04076 w_k1_AXP_s105
                                 + 0.787201 w_k1_DIS_s105
                                 + 0.819783 w_k1_FDX_s105
                                 + 1.06812 w_k1_IBM_s105
                                 + 1.08253 w_k1_INTC_s105
                                 + 0.932879 w_k1_KO_s105
                                 + 1.21499 w_k1_MAT_s105
                                 + 0.806773 w_k1_MSFT_s105
                                 + 0.951641 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.774116 w_k1_ADM_s105
                                 + 0.809136 w_k1_AXP_s105
                                 + 1.10245 w_k1_DIS_s105
                                 + 0.977954 w_k1_FDX_s105 + 1.028 w_k1_IBM_s105
                                 + 1.20372 w_k1_INTC_s105
                                 + 1.07888 w_k1_KO_s105
                                 + 0.830531 w_k1_MAT_s105
                                 + 0.998174 w_k1_MSFT_s105
                                 + 1.06737 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.14653 w_k1_ADM_s105
                                 + 1.19992 w_k1_AXP_s105
                                 + 1.21856 w_k1_DIS_s105
                                 + 1.37532 w_k1_FDX_s105
                                 + 0.721061 w_k1_IBM_s105
                                 + 1.47077 w_k1_INTC_s105
                                 + 0.907752 w_k1_KO_s105
                                 + 0.690826 w_k1_MAT_s105
                                 + 1.14522 w_k1_MSFT_s105
                                 + 1.20663 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.19037 w_k1_ADM_s105
                                 + 1.59118 w_k1_AXP_s105
                                 + 1.20577 w_k1_DIS_s105
                                 + 1.11783 w_k1_FDX_s105
                                 + 0.846073 w_k1_IBM_s105
                                 + 1.31288 w_k1_INTC_s105
                                 + 1.02773 w_k1_KO_s105 + 1.05516 w_k1_MAT_s105
                                 + 1.02325 w_k1_MSFT_s105
                                 + 1.32949 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.993032 w_k1_ADM_s106
                                 + 0.992203 w_k1_AXP_s106
                                 + 1.00464 w_k1_DIS_s106
                                 + 0.989713 w_k1_FDX_s106
                                 + 1.21305 w_k1_IBM_s106
                                 + 1.16138 w_k1_INTC_s106
                                 + 1.08021 w_k1_KO_s106 + 1.09084 w_k1_MAT_s106
                                 + 1.01358 w_k1_MSFT_s106
                                 + 1.00458 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.33998 w_k1_ADM_s106
                                 + 1.36695 w_k1_AXP_s106
                                 + 1.28676 w_k1_DIS_s106
                                 + 1.17301 w_k1_FDX_s106
                                 + 0.993931 w_k1_IBM_s106
                                 + 1.57355 w_k1_INTC_s106
                                 + 1.18572 w_k1_KO_s106 + 1.29028 w_k1_MAT_s106
                                 + 1.24201 w_k1_MSFT_s106
                                 + 1.15004 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.678635 w_k1_ADM_s106
                                 + 0.813234 w_k1_AXP_s106
                                 + 0.953598 w_k1_DIS_s106
                                 + 0.982067 w_k1_FDX_s106
                                 + 0.776938 w_k1_IBM_s106
                                 + 1.11052 w_k1_INTC_s106
                                 + 0.955334 w_k1_KO_s106
                                 + 0.759346 w_k1_MAT_s106
                                 + 0.972716 w_k1_MSFT_s106
                                 + 0.905331 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.769357 w_k1_ADM_s106
                                 + 0.812968 w_k1_AXP_s106
                                 + 0.592575 w_k1_DIS_s106
                                 + 0.78667 w_k1_FDX_s106
                                 + 0.734174 w_k1_IBM_s106
                                 + 0.996741 w_k1_INTC_s106
                                 + 0.75536 w_k1_KO_s106
                                 + 0.967491 w_k1_MAT_s106
                                 + 0.796156 w_k1_MSFT_s106
                                 + 0.879943 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.43307 w_k1_ADM_s106
                                 + 1.0545 w_k1_AXP_s106 + 1.12797 w_k1_DIS_s106
                                 + 1.13496 w_k1_FDX_s106
                                 + 0.810068 w_k1_IBM_s106
                                 + 1.17174 w_k1_INTC_s106
                                 + 0.916382 w_k1_KO_s106
                                 + 0.841907 w_k1_MAT_s106
                                 + 1.09916 w_k1_MSFT_s106
                                 + 0.988994 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.03048 w_k1_ADM_s106
                                 + 0.836807 w_k1_AXP_s106
                                 + 1.55237 w_k1_DIS_s106
                                 + 1.09189 w_k1_FDX_s106
                                 + 1.24016 w_k1_IBM_s106
                                 + 1.04354 w_k1_INTC_s106
                                 + 1.27929 w_k1_KO_s106 + 1.15765 w_k1_MAT_s106
                                 + 1.19715 w_k1_MSFT_s106
                                 + 1.18665 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.46749 w_k1_ADM_s106
                                 + 1.53557 w_k1_AXP_s106
                                 + 1.45636 w_k1_DIS_s106
                                 + 1.11601 w_k1_FDX_s106
                                 + 1.55262 w_k1_IBM_s106
                                 + 1.1699 w_k1_INTC_s106 + 1.29904 w_k1_KO_s106
                                 + 1.98251 w_k1_MAT_s106
                                 + 1.1134 w_k1_MSFT_s106 + 1.15231 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.01625 w_k1_ADM_s106
                                 + 0.766959 w_k1_AXP_s106
                                 + 0.800869 w_k1_DIS_s106
                                 + 0.808585 w_k1_FDX_s106
                                 + 0.754716 w_k1_IBM_s106
                                 + 1.04246 w_k1_INTC_s106
                                 + 0.84242 w_k1_KO_s106
                                 + 0.782602 w_k1_MAT_s106
                                 + 0.917798 w_k1_MSFT_s106
                                 + 0.892747 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.595646 w_k1_ADM_s107
                                 + 0.867823 w_k1_AXP_s107
                                 + 0.828593 w_k1_DIS_s107
                                 + 0.969969 w_k1_FDX_s107
                                 + 0.856261 w_k1_IBM_s107
                                 + 0.928838 w_k1_INTC_s107
                                 + 0.79017 w_k1_KO_s107
                                 + 0.754584 w_k1_MAT_s107
                                 + 0.925409 w_k1_MSFT_s107
                                 + 0.975125 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.0561 w_k1_ADM_s107
                                 + 1.16033 w_k1_AXP_s107
                                 + 1.48986 w_k1_DIS_s107
                                 + 1.32657 w_k1_FDX_s107
                                 + 0.757854 w_k1_IBM_s107
                                 + 0.90607 w_k1_INTC_s107
                                 + 1.03156 w_k1_KO_s107 + 1.04887 w_k1_MAT_s107
                                 + 1.18392 w_k1_MSFT_s107
                                 + 1.27981 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.11351 w_k1_ADM_s107
                                 + 1.17773 w_k1_AXP_s107
                                 + 1.21939 w_k1_DIS_s107
                                 + 1.17063 w_k1_FDX_s107
                                 + 0.953635 w_k1_IBM_s107
                                 + 0.902658 w_k1_INTC_s107
                                 + 1.15351 w_k1_KO_s107 + 1.07102 w_k1_MAT_s107
                                 + 1.03929 w_k1_MSFT_s107
                                 + 1.34855 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.960872 w_k1_ADM_s107
                                 + 1.59714 w_k1_AXP_s107 + 1.5439 w_k1_DIS_s107
                                 + 1.35673 w_k1_FDX_s107
                                 + 1.62651 w_k1_IBM_s107
                                 + 1.35863 w_k1_INTC_s107
                                 + 1.19356 w_k1_KO_s107 + 1.58382 w_k1_MAT_s107
                                 + 1.05477 w_k1_MSFT_s107
                                 + 1.30042 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.857232 w_k1_ADM_s107
                                 + 1.2516 w_k1_AXP_s107
                                 + 0.813205 w_k1_DIS_s107
                                 + 0.915769 w_k1_FDX_s107
                                 + 0.876839 w_k1_IBM_s107
                                 + 0.858137 w_k1_INTC_s107
                                 + 0.88352 w_k1_KO_s107 + 1.30104 w_k1_MAT_s107
                                 + 0.84323 w_k1_MSFT_s107
                                 + 1.06812 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.765482 w_k1_ADM_s107
                                 + 1.1966 w_k1_AXP_s107 + 1.11283 w_k1_DIS_s107
                                 + 1.00847 w_k1_FDX_s107 + 1.3487 w_k1_IBM_s107
                                 + 0.938027 w_k1_INTC_s107
                                 + 1.03684 w_k1_KO_s107 + 1.72558 w_k1_MAT_s107
                                 + 1.04113 w_k1_MSFT_s107
                                 + 1.15867 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.725411 w_k1_ADM_s107
                                 + 1.05066 w_k1_AXP_s107
                                 + 0.811264 w_k1_DIS_s107
                                 + 0.853768 w_k1_FDX_s107
                                 + 1.36667 w_k1_IBM_s107
                                 + 1.02592 w_k1_INTC_s107
                                 + 1.06418 w_k1_KO_s107 + 1.45277 w_k1_MAT_s107
                                 + 0.902203 w_k1_MSFT_s107
                                 + 1.05268 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.622478 w_k1_ADM_s107
                                 + 0.728367 w_k1_AXP_s107
                                 + 0.620605 w_k1_DIS_s107
                                 + 0.795871 w_k1_FDX_s107
                                 + 0.685813 w_k1_IBM_s107
                                 + 0.64998 w_k1_INTC_s107
                                 + 0.662151 w_k1_KO_s107
                                 + 0.923283 w_k1_MAT_s107
                                 + 0.780811 w_k1_MSFT_s107
                                 + 0.936684 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.04964 w_k1_ADM_s110
                                 + 0.737559 w_k1_AXP_s110
                                 + 0.816447 w_k1_DIS_s110
                                 + 0.889189 w_k1_FDX_s110
                                 + 0.927524 w_k1_IBM_s110
                                 + 0.754027 w_k1_INTC_s110
                                 + 1.1324 w_k1_KO_s110 + 0.962361 w_k1_MAT_s110
                                 + 0.830254 w_k1_MSFT_s110
                                 + 0.986731 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.56188 w_k1_ADM_s110
                                 + 1.5022 w_k1_AXP_s110 + 1.57124 w_k1_DIS_s110
                                 + 1.30062 w_k1_FDX_s110
                                 + 1.55745 w_k1_IBM_s110
                                 + 1.18547 w_k1_INTC_s110 + 1.5504 w_k1_KO_s110
                                 + 1.88393 w_k1_MAT_s110
                                 + 1.37916 w_k1_MSFT_s110 + 1.1211 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.07578 w_k1_ADM_s110
                                 + 0.828972 w_k1_AXP_s110
                                 + 0.672292 w_k1_DIS_s110
                                 + 0.912934 w_k1_FDX_s110
                                 + 0.686796 w_k1_IBM_s110
                                 + 0.719022 w_k1_INTC_s110
                                 + 0.804266 w_k1_KO_s110
                                 + 0.835143 w_k1_MAT_s110
                                 + 0.726663 w_k1_MSFT_s110
                                 + 0.804457 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.944853 w_k1_ADM_s110
                                 + 0.705557 w_k1_AXP_s110
                                 + 0.579142 w_k1_DIS_s110
                                 + 0.731245 w_k1_FDX_s110
                                 + 0.857948 w_k1_IBM_s110
                                 + 0.89908 w_k1_INTC_s110
                                 + 0.829699 w_k1_KO_s110
                                 + 0.867708 w_k1_MAT_s110
                                 + 0.782511 w_k1_MSFT_s110
                                 + 0.842129 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.26178 w_k1_ADM_s110
                                 + 0.832193 w_k1_AXP_s110
                                 + 0.854176 w_k1_DIS_s110
                                 + 0.868749 w_k1_FDX_s110
                                 + 1.28705 w_k1_IBM_s110
                                 + 0.707232 w_k1_INTC_s110
                                 + 1.15148 w_k1_KO_s110 + 1.31373 w_k1_MAT_s110
                                 + 0.851858 w_k1_MSFT_s110
                                 + 1.00316 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.2822 w_k1_ADM_s110
                                 + 1.14626 w_k1_AXP_s110
                                 + 1.11921 w_k1_DIS_s110
                                 + 1.18151 w_k1_FDX_s110
                                 + 1.10825 w_k1_IBM_s110
                                 + 0.973548 w_k1_INTC_s110
                                 + 1.24826 w_k1_KO_s110 + 1.31306 w_k1_MAT_s110
                                 + 1.06014 w_k1_MSFT_s110
                                 + 1.06735 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.20675 w_k1_ADM_s110
                                 + 0.981747 w_k1_AXP_s110
                                 + 0.812606 w_k1_DIS_s110
                                 + 0.997532 w_k1_FDX_s110
                                 + 0.995166 w_k1_IBM_s110
                                 + 1.01138 w_k1_INTC_s110
                                 + 0.990562 w_k1_KO_s110
                                 + 0.963382 w_k1_MAT_s110
                                 + 1.04731 w_k1_MSFT_s110
                                 + 1.01008 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.35593 w_k1_ADM_s110
                                 + 1.45815 w_k1_AXP_s110 + 1.533 w_k1_DIS_s110
                                 + 1.44367 w_k1_FDX_s110
                                 + 1.03438 w_k1_IBM_s110
                                 + 1.2617 w_k1_INTC_s110 + 1.31693 w_k1_KO_s110
                                 + 1.16788 w_k1_MAT_s110
                                 + 1.16238 w_k1_MSFT_s110
                                 + 1.02333 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.804875 w_k1_ADM_s111
                                 + 0.573124 w_k1_AXP_s111
                                 + 0.560056 w_k1_DIS_s111
                                 + 0.669898 w_k1_FDX_s111
                                 + 0.653657 w_k1_IBM_s111
                                 + 0.527799 w_k1_INTC_s111
                                 + 0.659717 w_k1_KO_s111
                                 + 0.834618 w_k1_MAT_s111
                                 + 0.659897 w_k1_MSFT_s111
                                 + 0.735044 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.18735 w_k1_ADM_s111
                                 + 1.55727 w_k1_AXP_s111
                                 + 1.14293 w_k1_DIS_s111 + 1.3139 w_k1_FDX_s111
                                 + 1.00167 w_k1_IBM_s111
                                 + 1.02977 w_k1_INTC_s111
                                 + 0.978973 w_k1_KO_s111 + 1.1111 w_k1_MAT_s111
                                 + 1.0238 w_k1_MSFT_s111 + 1.18475 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.24319 w_k1_ADM_s111
                                 + 1.06547 w_k1_AXP_s111
                                 + 1.21251 w_k1_DIS_s111
                                 + 1.14602 w_k1_FDX_s111
                                 + 0.915038 w_k1_IBM_s111
                                 + 0.930274 w_k1_INTC_s111
                                 + 1.02284 w_k1_KO_s111 + 1.22988 w_k1_MAT_s111
                                 + 1.12427 w_k1_MSFT_s111
                                 + 1.03798 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.06282 w_k1_ADM_s111
                                 + 1.21592 w_k1_AXP_s111
                                 + 1.01093 w_k1_DIS_s111
                                 + 0.938193 w_k1_FDX_s111
                                 + 1.10803 w_k1_IBM_s111
                                 + 0.954595 w_k1_INTC_s111
                                 + 1.15135 w_k1_KO_s111 + 1.21994 w_k1_MAT_s111
                                 + 0.999216 w_k1_MSFT_s111
                                 + 0.987377 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.845617 w_k1_ADM_s111
                                 + 0.837771 w_k1_AXP_s111
                                 + 0.760695 w_k1_DIS_s111
                                 + 0.923713 w_k1_FDX_s111
                                 + 0.701144 w_k1_IBM_s111
                                 + 0.830716 w_k1_INTC_s111
                                 + 0.895523 w_k1_KO_s111
                                 + 0.831584 w_k1_MAT_s111
                                 + 0.998823 w_k1_MSFT_s111
                                 + 0.99725 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.693939 w_k1_ADM_s111
                                 + 1.00479 w_k1_AXP_s111
                                 + 0.744315 w_k1_DIS_s111
                                 + 0.766931 w_k1_FDX_s111
                                 + 1.10005 w_k1_IBM_s111
                                 + 0.874621 w_k1_INTC_s111
                                 + 0.90576 w_k1_KO_s111 + 1.69541 w_k1_MAT_s111
                                 + 0.750937 w_k1_MSFT_s111
                                 + 0.922919 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.58765 w_k1_ADM_s111
                                 + 1.47305 w_k1_AXP_s111 + 1.467 w_k1_DIS_s111
                                 + 1.25532 w_k1_FDX_s111 + 1.5448 w_k1_IBM_s111
                                 + 1.04816 w_k1_INTC_s111
                                 + 1.40334 w_k1_KO_s111 + 1.7406 w_k1_MAT_s111
                                 + 1.19314 w_k1_MSFT_s111
                                 + 1.26963 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.57675 w_k1_ADM_s111
                                 + 1.54376 w_k1_AXP_s111
                                 + 1.58515 w_k1_DIS_s111
                                 + 1.41413 w_k1_FDX_s111 + 1.0923 w_k1_IBM_s111
                                 + 1.23472 w_k1_INTC_s111
                                 + 1.33675 w_k1_KO_s111 + 1.47533 w_k1_MAT_s111
                                 + 1.47234 w_k1_MSFT_s111
                                 + 1.26083 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.956799 w_k1_ADM_s112
                                 + 1.02499 w_k1_AXP_s112
                                 + 1.12412 w_k1_DIS_s112
                                 + 0.974273 w_k1_FDX_s112
                                 + 1.24437 w_k1_IBM_s112
                                 + 1.23052 w_k1_INTC_s112
                                 + 1.17707 w_k1_KO_s112 + 1.41469 w_k1_MAT_s112
                                 + 1.00176 w_k1_MSFT_s112
                                 + 0.998912 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.37992 w_k1_ADM_s112
                                 + 1.38415 w_k1_AXP_s112
                                 + 1.37298 w_k1_DIS_s112
                                 + 1.33242 w_k1_FDX_s112
                                 + 1.15898 w_k1_IBM_s112
                                 + 1.37988 w_k1_INTC_s112
                                 + 1.18609 w_k1_KO_s112 + 1.02813 w_k1_MAT_s112
                                 + 1.31105 w_k1_MSFT_s112
                                 + 1.14709 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.891903 w_k1_ADM_s112
                                 + 0.905927 w_k1_AXP_s112
                                 + 0.844085 w_k1_DIS_s112
                                 + 0.771789 w_k1_FDX_s112
                                 + 0.835452 w_k1_IBM_s112
                                 + 0.765479 w_k1_INTC_s112
                                 + 0.854871 w_k1_KO_s112
                                 + 1.00485 w_k1_MAT_s112
                                 + 0.90878 w_k1_MSFT_s112
                                 + 0.877195 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.889832 w_k1_ADM_s112
                                 + 1.2096 w_k1_AXP_s112 + 1.16605 w_k1_DIS_s112
                                 + 0.928062 w_k1_FDX_s112
                                 + 1.62061 w_k1_IBM_s112
                                 + 1.3026 w_k1_INTC_s112 + 1.25408 w_k1_KO_s112
                                 + 1.35246 w_k1_MAT_s112
                                 + 1.1537 w_k1_MSFT_s112 + 1.09654 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.885348 w_k1_ADM_s112
                                 + 0.728563 w_k1_AXP_s112
                                 + 0.792425 w_k1_DIS_s112
                                 + 0.700254 w_k1_FDX_s112
                                 + 1.08429 w_k1_IBM_s112
                                 + 0.83646 w_k1_INTC_s112
                                 + 0.963561 w_k1_KO_s112
                                 + 0.923194 w_k1_MAT_s112
                                 + 0.889265 w_k1_MSFT_s112
                                 + 0.870265 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.02521 w_k1_ADM_s112
                                 + 0.895769 w_k1_AXP_s112
                                 + 1.01358 w_k1_DIS_s112
                                 + 1.03109 w_k1_FDX_s112
                                 + 0.761763 w_k1_IBM_s112
                                 + 0.927469 w_k1_INTC_s112
                                 + 1.10199 w_k1_KO_s112 + 0.87491 w_k1_MAT_s112
                                 + 1.01475 w_k1_MSFT_s112
                                 + 1.06678 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.37733 w_k1_ADM_s112
                                 + 1.57234 w_k1_AXP_s112
                                 + 1.85851 w_k1_DIS_s112
                                 + 1.45048 w_k1_FDX_s112 + 1.2989 w_k1_IBM_s112
                                 + 1.2213 w_k1_INTC_s112 + 1.28065 w_k1_KO_s112
                                 + 1.50455 w_k1_MAT_s112
                                 + 1.2527 w_k1_MSFT_s112 + 1.28305 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.1027 w_k1_ADM_s112
                                 + 1.32077 w_k1_AXP_s112
                                 + 1.08011 w_k1_DIS_s112
                                 + 1.06693 w_k1_FDX_s112
                                 + 0.968021 w_k1_IBM_s112
                                 + 1.07073 w_k1_INTC_s112
                                 + 0.98536 w_k1_KO_s112 + 1.24056 w_k1_MAT_s112
                                 + 1.04511 w_k1_MSFT_s112
                                 + 1.01809 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.35064 w_k1_ADM_s113
                                 + 1.46117 w_k1_AXP_s113 + 1.7401 w_k1_DIS_s113
                                 + 1.38175 w_k1_FDX_s113
                                 + 1.04248 w_k1_IBM_s113 + 1.265 w_k1_INTC_s113
                                 + 1.22566 w_k1_KO_s113 + 1.31865 w_k1_MAT_s113
                                 + 1.46758 w_k1_MSFT_s113
                                 + 1.08468 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.01019 w_k1_ADM_s113
                                 + 1.02506 w_k1_AXP_s113
                                 + 0.991736 w_k1_DIS_s113
                                 + 1.04747 w_k1_FDX_s113
                                 + 0.759648 w_k1_IBM_s113
                                 + 1.25634 w_k1_INTC_s113
                                 + 0.928351 w_k1_KO_s113
                                 + 0.842985 w_k1_MAT_s113
                                 + 1.1143 w_k1_MSFT_s113
                                 + 0.940458 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.03034 w_k1_ADM_s113
                                 + 1.23536 w_k1_AXP_s113
                                 + 0.891559 w_k1_DIS_s113
                                 + 0.995481 w_k1_FDX_s113
                                 + 1.01677 w_k1_IBM_s113
                                 + 1.09189 w_k1_INTC_s113
                                 + 0.87041 w_k1_KO_s113 + 1.02352 w_k1_MAT_s113
                                 + 0.964081 w_k1_MSFT_s113
                                 + 0.91134 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.80878 w_k1_ADM_s113
                                 + 0.675343 w_k1_AXP_s113
                                 + 0.734337 w_k1_DIS_s113
                                 + 0.770994 w_k1_FDX_s113
                                 + 1.11264 w_k1_IBM_s113
                                 + 0.918892 w_k1_INTC_s113
                                 + 0.957406 w_k1_KO_s113
                                 + 0.938913 w_k1_MAT_s113
                                 + 0.935575 w_k1_MSFT_s113
                                 + 0.93858 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.894641 w_k1_ADM_s113
                                 + 0.718505 w_k1_AXP_s113
                                 + 0.557616 w_k1_DIS_s113
                                 + 0.684363 w_k1_FDX_s113
                                 + 0.77077 w_k1_IBM_s113
                                 + 0.840307 w_k1_INTC_s113
                                 + 0.705917 w_k1_KO_s113
                                 + 0.666481 w_k1_MAT_s113
                                 + 0.713941 w_k1_MSFT_s113
                                 + 0.725332 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.12192 w_k1_ADM_s113
                                 + 1.06183 w_k1_AXP_s113
                                 + 1.16471 w_k1_DIS_s113
                                 + 1.07434 w_k1_FDX_s113
                                 + 1.00872 w_k1_IBM_s113
                                 + 1.21254 w_k1_INTC_s113
                                 + 1.06692 w_k1_KO_s113
                                 + 0.910296 w_k1_MAT_s113
                                 + 1.20248 w_k1_MSFT_s113
                                 + 0.999107 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.55691 w_k1_ADM_s113
                                 + 1.46429 w_k1_AXP_s113
                                 + 1.49388 w_k1_DIS_s113
                                 + 1.30737 w_k1_FDX_s113
                                 + 1.36451 w_k1_IBM_s113
                                 + 1.49793 w_k1_INTC_s113
                                 + 1.48917 w_k1_KO_s113 + 1.3037 w_k1_MAT_s113
                                 + 1.49001 w_k1_MSFT_s113
                                 + 1.19099 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.16883 w_k1_ADM_s113
                                 + 1.48644 w_k1_AXP_s113
                                 + 1.08676 w_k1_DIS_s113
                                 + 1.05808 w_k1_FDX_s113
                                 + 1.28619 w_k1_IBM_s113
                                 + 1.38859 w_k1_INTC_s113
                                 + 1.17026 w_k1_KO_s113 + 1.08708 w_k1_MAT_s113
                                 + 1.05312 w_k1_MSFT_s113
                                 + 1.05724 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.05053 w_k1_ADM_s114
                                 + 0.949128 w_k1_AXP_s114
                                 + 0.889726 w_k1_DIS_s114
                                 + 0.869331 w_k1_FDX_s114
                                 + 1.07079 w_k1_IBM_s114
                                 + 1.26692 w_k1_INTC_s114
                                 + 0.897733 w_k1_KO_s114
                                 + 1.03469 w_k1_MAT_s114
                                 + 0.82461 w_k1_MSFT_s114
                                 + 0.971605 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.985928 w_k1_ADM_s114
                                 + 1.18141 w_k1_AXP_s114 + 1.4035 w_k1_DIS_s114
                                 + 1.16744 w_k1_FDX_s114
                                 + 1.33268 w_k1_IBM_s114
                                 + 1.14801 w_k1_INTC_s114
                                 + 1.08221 w_k1_KO_s114 + 1.15181 w_k1_MAT_s114
                                 + 1.02763 w_k1_MSFT_s114
                                 + 1.17358 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.18455 w_k1_ADM_s114
                                 + 1.17941 w_k1_AXP_s114 + 1.3924 w_k1_DIS_s114
                                 + 1.16647 w_k1_FDX_s114
                                 + 1.68315 w_k1_IBM_s114
                                 + 1.29901 w_k1_INTC_s114 + 1.4289 w_k1_KO_s114
                                 + 1.80295 w_k1_MAT_s114
                                 + 1.31033 w_k1_MSFT_s114
                                 + 1.12741 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.752886 w_k1_ADM_s114
                                 + 0.65908 w_k1_AXP_s114
                                 + 0.661832 w_k1_DIS_s114
                                 + 0.787701 w_k1_FDX_s114
                                 + 0.664791 w_k1_IBM_s114
                                 + 1.01726 w_k1_INTC_s114
                                 + 0.653053 w_k1_KO_s114
                                 + 0.643223 w_k1_MAT_s114
                                 + 0.790172 w_k1_MSFT_s114
                                 + 0.797639 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.19043 w_k1_ADM_s114
                                 + 1.56892 w_k1_AXP_s114 + 1.8934 w_k1_DIS_s114
                                 + 1.47033 w_k1_FDX_s114
                                 + 1.09767 w_k1_IBM_s114
                                 + 2.00532 w_k1_INTC_s114
                                 + 1.28983 w_k1_KO_s114 + 1.58895 w_k1_MAT_s114
                                 + 1.4859 w_k1_MSFT_s114 + 1.22198 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.05121 w_k1_ADM_s114
                                 + 1.23671 w_k1_AXP_s114
                                 + 1.53024 w_k1_DIS_s114
                                 + 1.21647 w_k1_FDX_s114
                                 + 1.03321 w_k1_IBM_s114
                                 + 1.67814 w_k1_INTC_s114
                                 + 1.34151 w_k1_KO_s114 + 1.21517 w_k1_MAT_s114
                                 + 1.37541 w_k1_MSFT_s114
                                 + 1.14485 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.600035 w_k1_ADM_s114
                                 + 0.889761 w_k1_AXP_s114
                                 + 0.890315 w_k1_DIS_s114
                                 + 0.759645 w_k1_FDX_s114
                                 + 1.45432 w_k1_IBM_s114
                                 + 1.14943 w_k1_INTC_s114
                                 + 0.87933 w_k1_KO_s114 + 1.58593 w_k1_MAT_s114
                                 + 0.963021 w_k1_MSFT_s114
                                 + 0.91426 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.846668 w_k1_ADM_s114
                                 + 0.901312 w_k1_AXP_s114
                                 + 1.21147 w_k1_DIS_s114
                                 + 0.98341 w_k1_FDX_s114
                                 + 0.838477 w_k1_IBM_s114
                                 + 1.07894 w_k1_INTC_s114
                                 + 0.905946 w_k1_KO_s114
                                 + 0.753683 w_k1_MAT_s114
                                 + 0.995115 w_k1_MSFT_s114
                                 + 0.95244 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.24535 w_k1_ADM_s115
                                 + 1.18416 w_k1_AXP_s115
                                 + 1.22782 w_k1_DIS_s115
                                 + 1.29302 w_k1_FDX_s115
                                 + 0.930446 w_k1_IBM_s115
                                 + 1.11989 w_k1_INTC_s115
                                 + 1.12905 w_k1_KO_s115 + 1.60211 w_k1_MAT_s115
                                 + 1.14634 w_k1_MSFT_s115
                                 + 1.13218 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.85998 w_k1_ADM_s115
                                 + 0.71184 w_k1_AXP_s115 + 0.7428 w_k1_DIS_s115
                                 + 0.987142 w_k1_FDX_s115
                                 + 0.397646 w_k1_IBM_s115
                                 + 0.861194 w_k1_INTC_s115
                                 + 0.68582 w_k1_KO_s115
                                 + 0.584043 w_k1_MAT_s115
                                 + 1.05409 w_k1_MSFT_s115
                                 + 0.84191 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.896254 w_k1_ADM_s115
                                 + 0.857198 w_k1_AXP_s115
                                 + 0.71403 w_k1_DIS_s115
                                 + 0.915981 w_k1_FDX_s115
                                 + 0.720595 w_k1_IBM_s115
                                 + 0.768183 w_k1_INTC_s115
                                 + 0.815107 w_k1_KO_s115
                                 + 0.855618 w_k1_MAT_s115
                                 + 0.934604 w_k1_MSFT_s115
                                 + 0.976868 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.14978 w_k1_ADM_s115
                                 + 0.890748 w_k1_AXP_s115
                                 + 1.06207 w_k1_DIS_s115
                                 + 0.927348 w_k1_FDX_s115
                                 + 1.25975 w_k1_IBM_s115
                                 + 0.924385 w_k1_INTC_s115
                                 + 1.26858 w_k1_KO_s115 + 1.69069 w_k1_MAT_s115
                                 + 1.02399 w_k1_MSFT_s115
                                 + 1.02362 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.782556 w_k1_ADM_s115
                                 + 0.611818 w_k1_AXP_s115
                                 + 0.64223 w_k1_DIS_s115
                                 + 0.739441 w_k1_FDX_s115
                                 + 0.681995 w_k1_IBM_s115
                                 + 0.655218 w_k1_INTC_s115
                                 + 0.835414 w_k1_KO_s115
                                 + 1.18595 w_k1_MAT_s115
                                 + 0.899627 w_k1_MSFT_s115
                                 + 0.808414 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.00709 w_k1_ADM_s115
                                 + 1.36461 w_k1_AXP_s115
                                 + 1.37424 w_k1_DIS_s115 + 1.108 w_k1_FDX_s115
                                 + 1.58187 w_k1_IBM_s115
                                 + 1.32858 w_k1_INTC_s115
                                 + 1.31901 w_k1_KO_s115 + 1.93025 w_k1_MAT_s115
                                 + 1.09077 w_k1_MSFT_s115
                                 + 1.23867 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.90895 w_k1_ADM_s115
                                 + 0.823185 w_k1_AXP_s115
                                 + 1.00901 w_k1_DIS_s115
                                 + 0.952283 w_k1_FDX_s115
                                 + 0.80959 w_k1_IBM_s115
                                 + 0.862517 w_k1_INTC_s115
                                 + 0.950442 w_k1_KO_s115
                                 + 0.865969 w_k1_MAT_s115
                                 + 1.15091 w_k1_MSFT_s115
                                 + 0.980582 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.0584 w_k1_ADM_s115
                                 + 0.918485 w_k1_AXP_s115
                                 + 0.741344 w_k1_DIS_s115
                                 + 0.870059 w_k1_FDX_s115
                                 + 1.06247 w_k1_IBM_s115
                                 + 0.814043 w_k1_INTC_s115
                                 + 0.868074 w_k1_KO_s115
                                 + 1.36609 w_k1_MAT_s115
                                 + 0.867303 w_k1_MSFT_s115
                                 + 0.868592 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.18895 w_k1_ADM_s116
                                 + 0.925269 w_k1_AXP_s116
                                 + 0.987812 w_k1_DIS_s116
                                 + 1.14397 w_k1_FDX_s116
                                 + 0.682838 w_k1_IBM_s116
                                 + 1.0289 w_k1_INTC_s116
                                 + 0.862668 w_k1_KO_s116
                                 + 0.508281 w_k1_MAT_s116
                                 + 1.07379 w_k1_MSFT_s116
                                 + 1.00601 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.952044 w_k1_ADM_s116
                                 + 1.03314 w_k1_AXP_s116
                                 + 1.12219 w_k1_DIS_s116
                                 + 1.21343 w_k1_FDX_s116
                                 + 0.965557 w_k1_IBM_s116
                                 + 0.9428 w_k1_INTC_s116
                                 + 0.952744 w_k1_KO_s116
                                 + 1.12013 w_k1_MAT_s116
                                 + 0.935055 w_k1_MSFT_s116
                                 + 1.00897 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.02148 w_k1_ADM_s116
                                 + 0.861686 w_k1_AXP_s116
                                 + 1.0193 w_k1_DIS_s116
                                 + 0.854297 w_k1_FDX_s116
                                 + 1.39845 w_k1_IBM_s116
                                 + 0.871941 w_k1_INTC_s116
                                 + 1.17613 w_k1_KO_s116 + 1.05887 w_k1_MAT_s116
                                 + 0.948416 w_k1_MSFT_s116
                                 + 1.09677 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.11536 w_k1_ADM_s116
                                 + 0.961747 w_k1_AXP_s116
                                 + 0.9984 w_k1_DIS_s116
                                 + 0.912891 w_k1_FDX_s116
                                 + 1.09349 w_k1_IBM_s116
                                 + 0.844053 w_k1_INTC_s116
                                 + 1.03046 w_k1_KO_s116
                                 + 0.810977 w_k1_MAT_s116
                                 + 0.917625 w_k1_MSFT_s116
                                 + 1.02129 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.44848 w_k1_ADM_s116
                                 + 1.56772 w_k1_AXP_s116 + 1.7831 w_k1_DIS_s116
                                 + 1.37448 w_k1_FDX_s116
                                 + 1.36036 w_k1_IBM_s116
                                 + 1.50727 w_k1_INTC_s116
                                 + 1.45568 w_k1_KO_s116 + 1.27243 w_k1_MAT_s116
                                 + 1.2857 w_k1_MSFT_s116 + 1.2142 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.02222 w_k1_ADM_s116
                                 + 0.854437 w_k1_AXP_s116
                                 + 0.882165 w_k1_DIS_s116
                                 + 0.954391 w_k1_FDX_s116
                                 + 0.850659 w_k1_IBM_s116
                                 + 0.850758 w_k1_INTC_s116
                                 + 0.924384 w_k1_KO_s116
                                 + 0.651438 w_k1_MAT_s116
                                 + 0.912702 w_k1_MSFT_s116
                                 + 0.959871 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.26762 w_k1_ADM_s116
                                 + 1.32705 w_k1_AXP_s116
                                 + 0.967911 w_k1_DIS_s116
                                 + 1.03461 w_k1_FDX_s116
                                 + 0.852452 w_k1_IBM_s116
                                 + 0.866338 w_k1_INTC_s116
                                 + 0.972283 w_k1_KO_s116
                                 + 1.51912 w_k1_MAT_s116
                                 + 0.907338 w_k1_MSFT_s116
                                 + 1.02431 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.19238 w_k1_ADM_s116
                                 + 0.899677 w_k1_AXP_s116
                                 + 1.11545 w_k1_DIS_s116
                                 + 1.05012 w_k1_FDX_s116
                                 + 0.772162 w_k1_IBM_s116
                                 + 0.935666 w_k1_INTC_s116
                                 + 0.963749 w_k1_KO_s116
                                 + 0.713036 w_k1_MAT_s116
                                 + 1.0807 w_k1_MSFT_s116
                                 + 0.930187 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.46559 w_k1_ADM_s117
                                 + 1.47188 w_k1_AXP_s117
                                 + 1.58947 w_k1_DIS_s117
                                 + 1.37985 w_k1_FDX_s117
                                 + 1.38309 w_k1_IBM_s117
                                 + 0.862749 w_k1_INTC_s117
                                 + 1.19348 w_k1_KO_s117 + 1.34762 w_k1_MAT_s117
                                 + 1.21393 w_k1_MSFT_s117
                                 + 1.22729 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.12407 w_k1_ADM_s117
                                 + 1.07684 w_k1_AXP_s117
                                 + 0.803484 w_k1_DIS_s117
                                 + 0.883933 w_k1_FDX_s117
                                 + 1.11292 w_k1_IBM_s117
                                 + 0.996854 w_k1_INTC_s117
                                 + 1.01567 w_k1_KO_s117 + 1.02665 w_k1_MAT_s117
                                 + 1.02374 w_k1_MSFT_s117
                                 + 0.89864 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.841341 w_k1_ADM_s117
                                 + 0.965782 w_k1_AXP_s117
                                 + 0.65319 w_k1_DIS_s117
                                 + 0.894565 w_k1_FDX_s117
                                 + 0.722968 w_k1_IBM_s117
                                 + 0.879088 w_k1_INTC_s117
                                 + 0.626716 w_k1_KO_s117
                                 + 0.761313 w_k1_MAT_s117
                                 + 0.864621 w_k1_MSFT_s117
                                 + 0.682498 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.08071 w_k1_ADM_s117
                                 + 0.928378 w_k1_AXP_s117
                                 + 1.08993 w_k1_DIS_s117
                                 + 1.02863 w_k1_FDX_s117
                                 + 0.937555 w_k1_IBM_s117
                                 + 1.00204 w_k1_INTC_s117
                                 + 1.00305 w_k1_KO_s117
                                 + 0.614535 w_k1_MAT_s117
                                 + 1.07898 w_k1_MSFT_s117
                                 + 0.901863 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.52134 w_k1_ADM_s117
                                 + 1.48263 w_k1_AXP_s117
                                 + 1.12194 w_k1_DIS_s117
                                 + 1.13492 w_k1_FDX_s117
                                 + 1.39976 w_k1_IBM_s117
                                 + 0.907371 w_k1_INTC_s117
                                 + 0.978675 w_k1_KO_s117
                                 + 1.41866 w_k1_MAT_s117
                                 + 1.0027 w_k1_MSFT_s117 + 1.01193 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.08943 w_k1_ADM_s117
                                 + 1.21544 w_k1_AXP_s117
                                 + 1.12576 w_k1_DIS_s117
                                 + 1.29924 w_k1_FDX_s117
                                 + 0.856967 w_k1_IBM_s117
                                 + 1.08173 w_k1_INTC_s117
                                 + 0.883147 w_k1_KO_s117
                                 + 0.827213 w_k1_MAT_s117
                                 + 1.19835 w_k1_MSFT_s117
                                 + 0.885635 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.892566 w_k1_ADM_s117
                                 + 0.662524 w_k1_AXP_s117
                                 + 0.63215 w_k1_DIS_s117
                                 + 0.778129 w_k1_FDX_s117
                                 + 0.742206 w_k1_IBM_s117
                                 + 0.642053 w_k1_INTC_s117
                                 + 0.64096 w_k1_KO_s117 + 0.64311 w_k1_MAT_s117
                                 + 0.766169 w_k1_MSFT_s117
                                 + 0.763029 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.94556 w_k1_ADM_s117
                                 + 0.954895 w_k1_AXP_s117
                                 + 1.16845 w_k1_DIS_s117
                                 + 0.968655 w_k1_FDX_s117
                                 + 1.30028 w_k1_IBM_s117
                                 + 1.11947 w_k1_INTC_s117
                                 + 1.00287 w_k1_KO_s117
                                 + 0.846937 w_k1_MAT_s117
                                 + 1.04798 w_k1_MSFT_s117
                                 + 0.938044 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0674780161307513 w_k1_ADM_s000
                                 + 0.0659928264910179 w_k1_ADM_s001
                                 + 0.0654171359992875 w_k1_ADM_s002
                                 + 0.124807247518357 w_k1_ADM_s003
                                 + 0.0757318575606662 w_k1_ADM_s004
                                 + 0.0733535457013802 w_k1_ADM_s005
                                 + 0.0419958125788105 w_k1_ADM_s006
                                 + 0.0629720665707388 w_k1_ADM_s007
                                 + 0.031217124504876 w_k1_ADM_s010
                                 + 0.016137467636779 w_k1_ADM_s011
                                 + 0.0701940827763096 w_k1_ADM_s012
                                 + 0.0245435389423747 w_k1_ADM_s013
                                 + 0.0474274592057536 w_k1_ADM_s014
                                 + 0.0156697805664926 w_k1_ADM_s015
                                 + 0.0148131080772668 w_k1_ADM_s016
                                 + 0.0317045977195793 w_k1_ADM_s017
                                 + 0.0145621158035189 w_k1_ADM_s100
                                 + 0.014238529232745 w_k1_ADM_s101
                                 + 0.00280148039437361 w_k1_ADM_s102
                                 + 0.00815556778397887 w_k1_ADM_s103
                                 + 0.00352237893659539 w_k1_ADM_s104
                                 + 0.00421512158942775 w_k1_ADM_s105
                                 + 0.00486296311455318 w_k1_ADM_s106
                                 + 0.00424504022859833 w_k1_ADM_s107
                                 + 0.00726122513189268 w_k1_ADM_s110
                                 + 0.0134037778026983 w_k1_ADM_s111
                                 + 0.0245584593162101 w_k1_ADM_s112
                                 + 0.0140121430416653 w_k1_ADM_s113
                                 + 0.0120807762945948 w_k1_ADM_s114
                                 + 0.0120717781112042 w_k1_ADM_s115
                                 + 0.0064365366047104 w_k1_ADM_s116
                                 + 0.0315692750106051 w_k1_ADM_s117
                                 + 0.0653208846988129 w_k1_AXP_s000
                                 + 0.0703313706823123 w_k1_AXP_s001
                                 + 0.068253813601747 w_k1_AXP_s002
                                 + 0.118482706544018 w_k1_AXP_s003
                                 + 0.0837255824276343 w_k1_AXP_s004
                                 + 0.0703628945913012 w_k1_AXP_s005
                                 + 0.0431415335533196 w_k1_AXP_s006
                                 + 0.072344802954984 w_k1_AXP_s007
                                 + 0.0294160752651589 w_k1_AXP_s010
                                 + 0.0149364144210432 w_k1_AXP_s011
                                 + 0.0624429181871233 w_k1_AXP_s012
                                 + 0.0238518877620627 w_k1_AXP_s013
                                 + 0.0452062049730995 w_k1_AXP_s014
                                 + 0.0157603774030367 w_k1_AXP_s015
                                 + 0.015075936433375 w_k1_AXP_s016
                                 + 0.0307306821714068 w_k1_AXP_s017
                                 + 0.0116530645061343 w_k1_AXP_s100
                                 + 0.0127624324415472 w_k1_AXP_s101
                                 + 0.00283526315835454 w_k1_AXP_s102
                                 + 0.00925957608269333 w_k1_AXP_s103
                                 + 0.00308372305272077 w_k1_AXP_s104
                                 + 0.00507355308002868 w_k1_AXP_s105
                                 + 0.00476052770265499 w_k1_AXP_s106
                                 + 0.00589534717877873 w_k1_AXP_s107
                                 + 0.00600797845934373 w_k1_AXP_s110
                                 + 0.0135475259173696 w_k1_AXP_s111
                                 + 0.0254213706542225 w_k1_AXP_s112
                                 + 0.0138020484646647 w_k1_AXP_s113
                                 + 0.0128547471737122 w_k1_AXP_s114
                                 + 0.0117797396796573 w_k1_AXP_s115
                                 + 0.00589430788774456 w_k1_AXP_s116
                                 + 0.0310133951763354 w_k1_AXP_s117
                                 + 0.0728851286882729 w_k1_DIS_s000
                                 + 0.0677044559496657 w_k1_DIS_s001
                                 + 0.0751899492545841 w_k1_DIS_s002
                                 + 0.114606262094517 w_k1_DIS_s003
                                 + 0.0882562549069349 w_k1_DIS_s004
                                 + 0.0739673943084556 w_k1_DIS_s005
                                 + 0.0398013726089755 w_k1_DIS_s006
                                 + 0.0747401761790262 w_k1_DIS_s007
                                 + 0.0323204712017697 w_k1_DIS_s010
                                 + 0.0157079300923096 w_k1_DIS_s011
                                 + 0.0683683525025349 w_k1_DIS_s012
                                 + 0.0237125653747566 w_k1_DIS_s013
                                 + 0.0490753692826695 w_k1_DIS_s014
                                 + 0.0172649319587665 w_k1_DIS_s015
                                 + 0.0153664575472382 w_k1_DIS_s016
                                 + 0.032283273714717 w_k1_DIS_s017
                                 + 0.0120000577551728 w_k1_DIS_s100
                                 + 0.0128038382775149 w_k1_DIS_s101
                                 + 0.00308435166604764 w_k1_DIS_s102
                                 + 0.00971486755238073 w_k1_DIS_s103
                                 + 0.00303804148598158 w_k1_DIS_s104
                                 + 0.00441453712022803 w_k1_DIS_s105
                                 + 0.00494113604106947 w_k1_DIS_s106
                                 + 0.00544490004481815 w_k1_DIS_s107
                                 + 0.00579869920770652 w_k1_DIS_s110
                                 + 0.0125933454098703 w_k1_DIS_s111
                                 + 0.0262826033459081 w_k1_DIS_s112
                                 + 0.0133968555674123 w_k1_DIS_s113
                                 + 0.0149123147746874 w_k1_DIS_s114
                                 + 0.0122114201245494 w_k1_DIS_s115
                                 + 0.00633426836702503 w_k1_DIS_s116
                                 + 0.0298490510697079 w_k1_DIS_s117
                                 + 0.0671454968213986 w_k1_FDX_s000
                                 + 0.0680046270447379 w_k1_FDX_s001
                                 + 0.0729868512864602 w_k1_FDX_s002
                                 + 0.112389900630986 w_k1_FDX_s003
                                 + 0.0822174993207009 w_k1_FDX_s004
                                 + 0.070671864836828 w_k1_FDX_s005
                                 + 0.0410006530241708 w_k1_FDX_s006
                                 + 0.0754957061361855 w_k1_FDX_s007
                                 + 0.0296225378729913 w_k1_FDX_s010
                                 + 0.0158740973112613 w_k1_FDX_s011
                                 + 0.069996437754786 w_k1_FDX_s012
                                 + 0.0251022530906193 w_k1_FDX_s013
                                 + 0.0464225155034316 w_k1_FDX_s014
                                 + 0.0165323755236152 w_k1_FDX_s015
                                 + 0.0155000943926831 w_k1_FDX_s016
                                 + 0.0329446332472727 w_k1_FDX_s017
                                 + 0.0126651245977816 w_k1_FDX_s100
                                 + 0.0128176416212635 w_k1_FDX_s101
                                 + 0.00291452646400823 w_k1_FDX_s102
                                 + 0.00976960572375501 w_k1_FDX_s103
                                 + 0.00324767646621572 w_k1_FDX_s104
                                 + 0.00439214812088755 w_k1_FDX_s105
                                 + 0.00468493399500673 w_k1_FDX_s106
                                 + 0.00539207891090423 w_k1_FDX_s107
                                 + 0.00614012462570684 w_k1_FDX_s110
                                 + 0.0125693804362381 w_k1_FDX_s111
                                 + 0.0237976870599432 w_k1_FDX_s112
                                 + 0.0130947696775801 w_k1_FDX_s113
                                 + 0.0129583015718714 w_k1_FDX_s114
                                 + 0.0120665370191313 w_k1_FDX_s115
                                 + 0.00609435129899926 w_k1_FDX_s116
                                 + 0.0303066633325871 w_k1_FDX_s117
                                 + 0.0709288036321985 w_k1_IBM_s000
                                 + 0.068730889274708 w_k1_IBM_s001
                                 + 0.0730317918388085 w_k1_IBM_s002
                                 + 0.119162182454773 w_k1_IBM_s003
                                 + 0.0835628230850607 w_k1_IBM_s004
                                 + 0.0740626275333314 w_k1_IBM_s005
                                 + 0.042228058228135 w_k1_IBM_s006
                                 + 0.0796123281847869 w_k1_IBM_s007
                                 + 0.0317777086996706 w_k1_IBM_s010
                                 + 0.0160361983277624 w_k1_IBM_s011
                                 + 0.0690367748311917 w_k1_IBM_s012
                                 + 0.0244432624524957 w_k1_IBM_s013
                                 + 0.0459274718956417 w_k1_IBM_s014
                                 + 0.0169242443848228 w_k1_IBM_s015
                                 + 0.0152138695882033 w_k1_IBM_s016
                                 + 0.0339502078383686 w_k1_IBM_s017
                                 + 0.0123160889406946 w_k1_IBM_s100
                                 + 0.0138202153563001 w_k1_IBM_s101
                                 + 0.0031767454449401 w_k1_IBM_s102
                                 + 0.0101096103100699 w_k1_IBM_s103
                                 + 0.00316682060556886 w_k1_IBM_s104
                                 + 0.00420733209161175 w_k1_IBM_s105
                                 + 0.00482911852941079 w_k1_IBM_s106
                                 + 0.00565438163426663 w_k1_IBM_s107
                                 + 0.00636479028910477 w_k1_IBM_s110
                                 + 0.0121329128828574 w_k1_IBM_s111
                                 + 0.0242202942630247 w_k1_IBM_s112
                                 + 0.0129184605125558 w_k1_IBM_s113
                                 + 0.0132582961196911 w_k1_IBM_s114
                                 + 0.0120262115100113 w_k1_IBM_s115
                                 + 0.00580295321868364 w_k1_IBM_s116
                                 + 0.0296318905520533 w_k1_IBM_s117
                                 + 0.0690722701194341 w_k1_INTC_s000
                                 + 0.067894945547356 w_k1_INTC_s001
                                 + 0.0772679249397559 w_k1_INTC_s002
                                 + 0.11819247631314 w_k1_INTC_s003
                                 + 0.0847248994583108 w_k1_INTC_s004
                                 + 0.0775192882261036 w_k1_INTC_s005
                                 + 0.0482277808109576 w_k1_INTC_s006
                                 + 0.070410785892132 w_k1_INTC_s007
                                 + 0.032076390391442 w_k1_INTC_s010
                                 + 0.0147932797108036 w_k1_INTC_s011
                                 + 0.0685793558383977 w_k1_INTC_s012
                                 + 0.026635639132514 w_k1_INTC_s013
                                 + 0.048120670207946 w_k1_INTC_s014
                                 + 0.0163439303460592 w_k1_INTC_s015
                                 + 0.0151179479713613 w_k1_INTC_s016
                                 + 0.0306390140947356 w_k1_INTC_s017
                                 + 0.0101167584862444 w_k1_INTC_s100
                                 + 0.0138001780727441 w_k1_INTC_s101
                                 + 0.00332182495893768 w_k1_INTC_s102
                                 + 0.0106379627401639 w_k1_INTC_s103
                                 + 0.00272367538062658 w_k1_INTC_s104
                                 + 0.00538281329003622 w_k1_INTC_s105
                                 + 0.00540659168435633 w_k1_INTC_s106
                                 + 0.00496958412635548 w_k1_INTC_s107
                                 + 0.00544700985020147 w_k1_INTC_s110
                                 + 0.0111870623727534 w_k1_INTC_s111
                                 + 0.0241135946057791 w_k1_INTC_s112
                                 + 0.0151285742220726 w_k1_INTC_s113
                                 + 0.0162099645946925 w_k1_INTC_s114
                                 + 0.011574551273571 w_k1_INTC_s115
                                 + 0.00558372859550344 w_k1_INTC_s116
                                 + 0.0269716182031497 w_k1_INTC_s117
                                 + 0.0696189043604968 w_k1_KO_s000
                                 + 0.0613628225521224 w_k1_KO_s001
                                 + 0.0643577725023587 w_k1_KO_s002
                                 + 0.120225798055336 w_k1_KO_s003
                                 + 0.0906996910949323 w_k1_KO_s004
                                 + 0.0702802989409462 w_k1_KO_s005
                                 + 0.0437401433862063 w_k1_KO_s006
                                 + 0.07093497360435 w_k1_KO_s007
                                 + 0.0325871852874506 w_k1_KO_s010
                                 + 0.0143901335624777 w_k1_KO_s011
                                 + 0.0650571079023119 w_k1_KO_s012
                                 + 0.0241326091761682 w_k1_KO_s013
                                 + 0.0506596091162109 w_k1_KO_s014
                                 + 0.0183544177184051 w_k1_KO_s015
                                 + 0.0152418253825546 w_k1_KO_s016
                                 + 0.0319410718202955 w_k1_KO_s017
                                 + 0.0123065982769455 w_k1_KO_s100
                                 + 0.014310070712037 w_k1_KO_s101
                                 + 0.00299277939387774 w_k1_KO_s102
                                 + 0.0100526277594558 w_k1_KO_s103
                                 + 0.00321836353283744 w_k1_KO_s104
                                 + 0.00421263807344356 w_k1_KO_s105
                                 + 0.0048556172906957 w_k1_KO_s106
                                 + 0.0050521499964951 w_k1_KO_s107
                                 + 0.00673736496099086 w_k1_KO_s110
                                 + 0.0127311079749883 w_k1_KO_s111
                                 + 0.0250894361408501 w_k1_KO_s112
                                 + 0.0132704988773585 w_k1_KO_s113
                                 + 0.0127432666079639 w_k1_KO_s114
                                 + 0.0123953226574661 w_k1_KO_s115
                                 + 0.00594634133723421 w_k1_KO_s116
                                 + 0.0263395859382741 w_k1_KO_s117
                                 + 0.0575866894077606 w_k1_MAT_s000
                                 + 0.065351008930013 w_k1_MAT_s001
                                 + 0.0684641590044786 w_k1_MAT_s002
                                 + 0.126921377188463 w_k1_MAT_s003
                                 + 0.0800636637106477 w_k1_MAT_s004
                                 + 0.0808128266502885 w_k1_MAT_s005
                                 + 0.0372378119344924 w_k1_MAT_s006
                                 + 0.0707474318918814 w_k1_MAT_s007
                                 + 0.0298986489614065 w_k1_MAT_s010
                                 + 0.0183170797810354 w_k1_MAT_s011
                                 + 0.0666613242796378 w_k1_MAT_s012
                                 + 0.0211784718362429 w_k1_MAT_s013
                                 + 0.0420303093471577 w_k1_MAT_s014
                                 + 0.0168071526789168 w_k1_MAT_s015
                                 + 0.0140581470735354 w_k1_MAT_s016
                                 + 0.0387457596204613 w_k1_MAT_s017
                                 + 0.0123629946641595 w_k1_MAT_s100
                                 + 0.0136989457797981 w_k1_MAT_s101
                                 + 0.00249759165899688 w_k1_MAT_s102
                                 + 0.0105117552194193 w_k1_MAT_s103
                                 + 0.00248268166001415 w_k1_MAT_s104
                                 + 0.00437923847073149 w_k1_MAT_s105
                                 + 0.00520344780400486 w_k1_MAT_s106
                                 + 0.00641326062394566 w_k1_MAT_s107
                                 + 0.00700545589018354 w_k1_MAT_s110
                                 + 0.0149458917446255 w_k1_MAT_s111
                                 + 0.0261095191410428 w_k1_MAT_s112
                                 + 0.0124332197207623 w_k1_MAT_s113
                                 + 0.0135257843242644 w_k1_MAT_s114
                                 + 0.0160467245708036 w_k1_MAT_s115
                                 + 0.00531698555157724 w_k1_MAT_s116
                                 + 0.0258484096004785 w_k1_MAT_s117
                                 + 0.0680726121559465 w_k1_MSFT_s000
                                 + 0.0674688509663358 w_k1_MSFT_s001
                                 + 0.0756367237220083 w_k1_MSFT_s002
                                 + 0.114934212668019 w_k1_MSFT_s003
                                 + 0.0822483069247697 w_k1_MSFT_s004
                                 + 0.0741188304925555 w_k1_MSFT_s005
                                 + 0.0397221090887502 w_k1_MSFT_s006
                                 + 0.0761155386673785 w_k1_MSFT_s007
                                 + 0.0336185530498222 w_k1_MSFT_s010
                                 + 0.016355967482415 w_k1_MSFT_s011
                                 + 0.0678934841309608 w_k1_MSFT_s012
                                 + 0.0248255607928882 w_k1_MSFT_s013
                                 + 0.0498616938459111 w_k1_MSFT_s014
                                 + 0.0189522435178312 w_k1_MSFT_s015
                                 + 0.0146598983526481 w_k1_MSFT_s016
                                 + 0.0314559772010323 w_k1_MSFT_s017
                                 + 0.0114853867951042 w_k1_MSFT_s100
                                 + 0.0122370589007087 w_k1_MSFT_s101
                                 + 0.00280604673569763 w_k1_MSFT_s102
                                 + 0.00967384102476139 w_k1_MSFT_s103
                                 + 0.00275702564092604 w_k1_MSFT_s104
                                 + 0.00413919275736686 w_k1_MSFT_s105
                                 + 0.00480120226712067 w_k1_MSFT_s106
                                 + 0.00495377276525478 w_k1_MSFT_s107
                                 + 0.00584834083866977 w_k1_MSFT_s110
                                 + 0.0125585511306673 w_k1_MSFT_s111
                                 + 0.0244044539323976 w_k1_MSFT_s112
                                 + 0.0142605294587111 w_k1_MSFT_s113
                                 + 0.0132455254289384 w_k1_MSFT_s114
                                 + 0.0124892639637333 w_k1_MSFT_s115
                                 + 0.00571728018762675 w_k1_MSFT_s116
                                 + 0.0296487014080482 w_k1_MSFT_s117
                                 + 0.0666981064705436 w_k1_PG_s000
                                 + 0.0676556661741324 w_k1_PG_s001
                                 + 0.0715524764206164 w_k1_PG_s002
                                 + 0.12659934873986 w_k1_PG_s003
                                 + 0.0818342504740428 w_k1_PG_s004
                                 + 0.0723912115296546 w_k1_PG_s005
                                 + 0.0377754403278951 w_k1_PG_s006
                                 + 0.0754752740818062 w_k1_PG_s007
                                 + 0.03199027365948 w_k1_PG_s010
                                 + 0.0153714343788955 w_k1_PG_s011
                                 + 0.0677154699113154 w_k1_PG_s012
                                 + 0.0231106785519199 w_k1_PG_s013
                                 + 0.0430237133450709 w_k1_PG_s014
                                 + 0.0184179404049248 w_k1_PG_s015
                                 + 0.0145284446556106 w_k1_PG_s016
                                 + 0.0328977551543166 w_k1_PG_s017
                                 + 0.0126530987980237 w_k1_PG_s100
                                 + 0.0141500092062763 w_k1_PG_s101
                                 + 0.00273462206832079 w_k1_PG_s102
                                 + 0.0100453047585238 w_k1_PG_s103
                                 + 0.00304049230424029 w_k1_PG_s104
                                 + 0.00467480437283801 w_k1_PG_s105
                                 + 0.00472423393254274 w_k1_PG_s106
                                 + 0.00583726796064416 w_k1_PG_s107
                                 + 0.00591457928547148 w_k1_PG_s110
                                 + 0.0127064108566762 w_k1_PG_s111
                                 + 0.0241370213103779 w_k1_PG_s112
                                 + 0.0124829610872427 w_k1_PG_s113
                                 + 0.0128813315797529 w_k1_PG_s114
                                 + 0.0122453933332982 w_k1_PG_s115
                                 + 0.00590239297874363 w_k1_PG_s116
                                 + 0.0260964002212482 w_k1_PG_s117 >= 1.2
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
