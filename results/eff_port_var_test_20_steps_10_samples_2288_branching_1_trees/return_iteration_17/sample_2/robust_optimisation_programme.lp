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
 w_k1_ADM_s0_return_constraint:  1.04435 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.965993 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.19751 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.891591 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.949366 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.983423 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.35981 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.981287 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.773829 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.813463 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.33135 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.0432 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.11841 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.09873 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.0582 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.988336 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.976565 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.18597 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.932249 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.36074 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.47673 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.037 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.913348 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.921499 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.10589 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.28921 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.880142 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.20021 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.794404 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.841394 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.993578 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.951371 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.12219 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.46948 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.14004 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.01854 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.780167 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.661333 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24629 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.902669 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.18717 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.791958 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.15782 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.926268 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.30782 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.20004 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.598288 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.909278 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.54518 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.12763 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.759215 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.951364 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.837048 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.09686 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.743059 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.16337 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.600805 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.20593 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.48309 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.85991 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.771315 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.939526 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.01873 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.1599 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.870214 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.40287 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.566811 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.697367 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.03521 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.959004 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.35855 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.9614 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.2442 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.21212 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.684463 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.766983 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34318 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.882582 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.16503 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.72262 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.13355 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.990039 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.38181 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.88027 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.620296 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.798628 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.98873 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.30064 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.998457 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.880103 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.815116 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.971902 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.681824 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.302 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.537418 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.21613 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.60105 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.02333 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.727816 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.713577 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.15805 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.36462 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.910298 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.80639 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.455719 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.797089 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.863308 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.952223 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.46722 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.94903 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.18695 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.23759 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.625531 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.559869 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19751 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.91884 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.08753 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.877356 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.985921 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.00738 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.3113 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.932999 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.726037 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.817225 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.45246 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.03715 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.983823 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.05661 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.900307 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.901152 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.73899 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.04669 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.698335 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.28044 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.38458 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.05452 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.831875 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.846651 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.06987 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.32325 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.839297 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.34335 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.62601 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.939781 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.971619 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.881929 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.42466 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.6571 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.23502 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.07846 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.806982 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.643317 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17265 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.954234 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.14231 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.736034 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.35072 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.841653 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.986841 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.80726 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.832295 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.18983 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.56195 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.43652 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.666501 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.70576 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.773561 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.52899 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.11127 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.21971 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.638673 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.783506 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.15625 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.607425 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.833828 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.05183 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.13569 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.832121 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.49238 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.65964 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.716209 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.63571 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.889742 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27057 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.86485 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.36132 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.845655 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.41089 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.689925 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.05423 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16576 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.919486 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.859288 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.03753 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.972188 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.20929 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.01019 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.905671 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.794355 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.37315 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.20567 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.052 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.890733 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.07526 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.07004 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.927344 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.05057 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.810776 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.17998 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.16199 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.16194 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.811688 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.871691 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.959296 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.16555 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.773084 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.32938 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.683318 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.918508 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.877383 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.05668 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.4698 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.61248 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.0486 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.08471 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.900783 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.80871 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15842 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.94587 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.17002 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.807942 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.17673 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.872129 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.15933 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.22263 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.782679 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.89224 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.47235 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.39466 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.938198 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.810277 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.842245 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.1227 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.869532 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.22994 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.653703 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.03841 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.20987 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.83963 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.768352 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.834928 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.10864 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.10167 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.01232 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.73815 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.635406 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.746684 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.966197 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.17987 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.25792 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.56297 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.04855 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.33032 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.737574 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.838897 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.31824 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.841882 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.14193 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.76456 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.3887 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.822397 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.873014 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 2.02423 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.643656 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.25915 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.49326 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.39734 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.504227 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.72138 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.548842 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.08009 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.02277 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.17489 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.477836 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.84527 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.10927 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.532956 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.95227 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.09033 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.41424 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.02927 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.62546 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.80313 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.668351 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.623159 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.20423 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.42505 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.11416 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.75851 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.05071 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.39981 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.667129 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.24213 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15338 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.893848 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.09635 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.832215 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.02052 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0798 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.14025 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.854368 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.769589 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.779461 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.34104 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.10282 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08482 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.0327 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.0519 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.954844 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.900181 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.12915 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.835858 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.32643 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.41812 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.11932 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.838315 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.88201 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06375 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.30885 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.865587 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.35832 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.630091 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.03682 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.90456 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.942549 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.32582 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.46509 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.10346 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.07588 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.833301 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.732582 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14167 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.929578 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.05625 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.818635 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.07871 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.975823 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.11002 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08902 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.867538 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.930178 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.14268 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.07758 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.945006 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.981894 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.969858 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.09854 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.90966 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.08641 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.843668 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.12415 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.31343 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.907148 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.931034 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.914216 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.12029 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.09025 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.03316 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.23576 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.771328 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.905587 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.98407 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.998823 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.22715 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.41027 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.02991 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.14089 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.826374 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.816355 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.549001256810766 z_k1_s0000
                                 - 0.357999807878466 z_k1_s0001
                                 - 0.339999847638765 z_k1_s0002
                                 - 0.82299900523739 z_k1_s0003
                                 - 0.297000017455033 z_k1_s0004
                                 - 0.437000007541581 z_k1_s0005
                                 - 0.727999589250066 z_k1_s0006
                                 - 0.957999829535098 z_k1_s0007
                                 - 0.0840000016706277 z_k1_s0010
                                 - 0.16800062654099 z_k1_s0011
                                 - 0.0869999293946094 z_k1_s0012
                                 - 0.127000004380591 z_k1_s0013
                                 - 0.280999202477084 z_k1_s0014
                                 - 0.340999314956057 z_k1_s0015
                                 - 0.242999598640203 z_k1_s0016
                                 - 0.227000503445413 z_k1_s0017
                                 - 0.234999852163056 z_k1_s0020
                                 - 0.803998702825379 z_k1_s0021
                                 - 0.587998722439929 z_k1_s0022
                                 - 0.901997020900813 z_k1_s0023
                                 - 0.185999811726606 z_k1_s0024
                                 - 0.381999532773038 z_k1_s0025
                                 - 0.587998722439929 z_k1_s0026
                                 - 0.721997771745216 z_k1_s0027
                                 - 0.587003507464308 z_k1_s0030
                                 - 0.996004263138658 z_k1_s0031
                                 - 0.410000912352115 z_k1_s0032
                                 - 0.359000656202165 z_k1_s0033
                                 - 0.564003287682742 z_k1_s0034
                                 - 0.491000068022965 z_k1_s0035
                                 - 0.316000686672328 z_k1_s0036
                                 - 1.11100052803645 z_k1_s0037
                                 - 0.133000046344982 z_k1_s0040
                                 - 0.241999502272794 z_k1_s0041
                                 - 0.130999997143551 z_k1_s0042
                                 - 0.149999944307117 z_k1_s0043
                                 - 0.293000652859273 z_k1_s0044
                                 - 0.550000003892771 z_k1_s0045
                                 - 0.470998996886304 z_k1_s0046
                                 - 0.110999921329264 z_k1_s0047
                                 - 0.217999973619396 z_k1_s0050
                                 - 0.229998955002456 z_k1_s0051
                                 - 0.267999120281009 z_k1_s0052
                                 - 0.462000445334516 z_k1_s0053
                                 - 0.534000277230883 z_k1_s0054
                                 - 0.158999933571043 z_k1_s0055
                                 - 0.205999849236719 z_k1_s0056
                                 - 0.20899999463235 z_k1_s0057
                                 - 0.1679999400799 z_k1_s0060
                                 - 0.215999857388454 z_k1_s0061
                                 - 0.242000826222091 z_k1_s0062
                                 - 0.130000024668959 z_k1_s0063
                                 - 0.391999707186459 z_k1_s0064
                                 - 0.160000030361796 z_k1_s0065
                                 - 0.63400053340855 z_k1_s0066
                                 - 0.353000778435598 z_k1_s0067
                                 - 0.288000721522971 z_k1_s0070
                                 - 0.285999646491118 z_k1_s0071
                                 - 0.375999907767015 z_k1_s0072
                                 - 0.475999243407744 z_k1_s0073
                                 - 0.371998903302297 z_k1_s0074
                                 - 0.224999937190375 z_k1_s0075
                                 - 0.495998537736395 z_k1_s0076
                                 - 0.34499999915849 z_k1_s0077
                                 - 0.459998719850232 z_k1_s0100
                                 - 0.187999707807944 z_k1_s0101
                                 - 0.386000311784897 z_k1_s0102
                                 - 0.20299966606393 z_k1_s0103
                                 - 0.107999699443098 z_k1_s0104
                                 - 0.358998769927006 z_k1_s0105
                                 - 0.217999624319915 z_k1_s0106
                                 - 0.387998458222324 z_k1_s0107
                                 - 0.164000098171078 z_k1_s0110
                                 - 0.124999751802891 z_k1_s0111
                                 - 0.373000119383175 z_k1_s0112
                                 - 0.193999530532629 z_k1_s0113
                                 - 0.0819998941858004 z_k1_s0114
                                 - 0.224999615205099 z_k1_s0115
                                 - 0.209999124525631 z_k1_s0116
                                 - 0.176000180915175 z_k1_s0117
                                 - 0.07799999539813 z_k1_s0120
                                 - 0.224000243440916 z_k1_s0121
                                 - 0.379999939837024 z_k1_s0122
                                 - 0.210999532740861 z_k1_s0123
                                 - 0.189000481403415 z_k1_s0124
                                 - 0.125000096420134 z_k1_s0125
                                 - 0.152000270334961 z_k1_s0126
                                 - 0.113000051835783 z_k1_s0127
                                 - 0.211999990309169 z_k1_s0130
                                 - 0.26799985403619 z_k1_s0131
                                 - 0.443000693682457 z_k1_s0132
                                 - 0.484000340811305 z_k1_s0133
                                 - 0.174999827246807 z_k1_s0134
                                 - 0.485000085277994 z_k1_s0135
                                 - 0.340998917095034 z_k1_s0136
                                 - 0.122999953786002 z_k1_s0137
                                 - 0.0920002416898327 z_k1_s0140
                                 - 0.0720003620184037 z_k1_s0141
                                 - 0.0250000483896011 z_k1_s0142
                                 - 0.0390000777597813 z_k1_s0143
                                 - 0.106000327860103 z_k1_s0144
                                 - 0.0550000950971044 z_k1_s0145
                                 - 0.110000190194209 z_k1_s0146
                                 - 0.0690001812673745 z_k1_s0147
                                 - 0.290000425028572 z_k1_s0150
                                 - 0.262999104185584 z_k1_s0151
                                 - 0.276000034517151 z_k1_s0152
                                 - 0.308000643257774 z_k1_s0153
                                 - 0.0969998955028626 z_k1_s0154
                                 - 0.475999368399712 z_k1_s0155
                                 - 0.156999894367468 z_k1_s0156
                                 - 0.11999999772921 z_k1_s0157
                                 - 0.0620000649268357 z_k1_s0160
                                 - 0.238000182009441 z_k1_s0161
                                 - 0.0789999886131914 z_k1_s0162
                                 - 0.284000331019712 z_k1_s0163
                                 - 0.0700000228851184 z_k1_s0164
                                 - 0.112000453416344 z_k1_s0165
                                 - 0.108000213937106 z_k1_s0166
                                 - 0.0890000663110929 z_k1_s0167
                                 - 0.149999677987135 z_k1_s0170
                                 - 0.0809996773930432 z_k1_s0171
                                 - 0.109000275257352 z_k1_s0172
                                 - 0.0420000179964051 z_k1_s0173
                                 - 0.0530000259287971 z_k1_s0174
                                 - 0.148000069853999 z_k1_s0175
                                 - 0.0510000121956341 z_k1_s0176
                                 - 0.0420000179964051 z_k1_s0177
                                 - 0.211999724554131 z_k1_s1000
                                 - 0.535997562683579 z_k1_s1001
                                 - 0.550000396949632 z_k1_s1002
                                 - 0.381999575693626 z_k1_s1003
                                 - 0.174999583194102 z_k1_s1004
                                 - 0.479000484501324 z_k1_s1005
                                 - 0.295999471383401 z_k1_s1006
                                 - 0.688999851574499 z_k1_s1007
                                 - 0.600997843346269 z_k1_s1010
                                 - 0.476999510872854 z_k1_s1011
                                 - 0.167999685540431 z_k1_s1012
                                 - 0.309999285176086 z_k1_s1013
                                 - 0.347999169191281 z_k1_s1014
                                 - 0.392998412105348 z_k1_s1015
                                 - 0.542999028478797 z_k1_s1016
                                 - 0.299999348750964 z_k1_s1017
                                 - 0.993001292485447 z_k1_s1020
                                 - 0.503999603006768 z_k1_s1021
                                 - 0.892999311311186 z_k1_s1022
                                 - 0.328999578998995 z_k1_s1023
                                 - 0.34099992267982 z_k1_s1024
                                 - 0.904001171309011 z_k1_s1025
                                 - 0.484999809253182 z_k1_s1026
                                 - 0.848001333130385 z_k1_s1027
                                 - 0.533999999495963 z_k1_s1030
                                 - 1.13599865199511 z_k1_s1031
                                 - 1.19899559475305 z_k1_s1032
                                 - 0.274999423749505 z_k1_s1033
                                 - 0.358998817550536 z_k1_s1034
                                 - 0.287999138400191 z_k1_s1035
                                 - 0.390998879550124 z_k1_s1036
                                 - 0.549998847499009 z_k1_s1037
                                 - 0.338000672470478 z_k1_s1040
                                 - 0.152999999905345 z_k1_s1041
                                 - 0.14099990265789 z_k1_s1042
                                 - 0.274999386921513 z_k1_s1043
                                 - 0.284000234856927 z_k1_s1044
                                 - 0.178000409170831 z_k1_s1045
                                 - 0.0969999077504689 z_k1_s1046
                                 - 0.0839999057323681 z_k1_s1047
                                 - 0.497996930071187 z_k1_s1050
                                 - 0.562996979636838 z_k1_s1051
                                 - 0.302999419363259 z_k1_s1052
                                 - 0.562996979636838 z_k1_s1053
                                 - 0.117999623259585 z_k1_s1054
                                 - 0.152999406442103 z_k1_s1055
                                 - 0.261999002546529 z_k1_s1056
                                 - 0.177999364629145 z_k1_s1057
                                 - 0.164999859218628 z_k1_s1060
                                 - 0.0790000386500962 z_k1_s1061
                                 - 0.357999951310636 z_k1_s1062
                                 - 0.372999352421611 z_k1_s1063
                                 - 0.138000016219172 z_k1_s1064
                                 - 0.271000477667939 z_k1_s1065
                                 - 0.236000647076063 z_k1_s1066
                                 - 0.221999609639673 z_k1_s1067
                                 - 0.211000104768581 z_k1_s1070
                                 - 0.203999516307694 z_k1_s1071
                                 - 0.355999188615353 z_k1_s1072
                                 - 0.0859999857844119 z_k1_s1073
                                 - 0.197000595845011 z_k1_s1074
                                 - 0.0809999943690308 z_k1_s1075
                                 - 0.265999009538851 z_k1_s1076
                                 - 0.266999808461065 z_k1_s1077
                                 - 1.07300134328237 z_k1_s1100
                                 - 0.737002233738339 z_k1_s1101
                                 - 0.701002104343739 z_k1_s1102
                                 - 0.837999296564442 z_k1_s1103
                                 - 2.03300217113141 z_k1_s1104
                                 - 1.55600340716079 z_k1_s1105
                                 - 0.422001298236115 z_k1_s1106
                                 - 0.508001126521939 z_k1_s1107
                                 - 0.398000053068699 z_k1_s1110
                                 - 0.255000692403969 z_k1_s1111
                                 - 0.304999189285465 z_k1_s1112
                                 - 0.314001313266524 z_k1_s1113
                                 - 0.102000051061362 z_k1_s1114
                                 - 0.313000575268128 z_k1_s1115
                                 - 0.216000147994689 z_k1_s1116
                                 - 0.356000683167612 z_k1_s1117
                                 - 0.148000176858695 z_k1_s1120
                                 - 0.151000141924018 z_k1_s1121
                                 - 0.145000211793372 z_k1_s1122
                                 - 0.169000300116491 z_k1_s1123
                                 - 0.415000377877251 z_k1_s1124
                                 - 0.1660001511509 z_k1_s1125
                                 - 0.258001041423657 z_k1_s1126
                                 - 0.38699972313281 z_k1_s1127
                                 - 0.350001393692458 z_k1_s1130
                                 - 0.191000130386017 z_k1_s1131
                                 - 0.632999369733023 z_k1_s1132
                                 - 0.235000226821585 z_k1_s1133
                                 - 0.454001020103383 z_k1_s1134
                                 - 0.248000042272855 z_k1_s1135
                                 - 0.36899989392038 z_k1_s1136
                                 - 0.277001496865448 z_k1_s1137
                                 - 0.480002743045237 z_k1_s1140
                                 - 0.689001910484582 z_k1_s1141
                                 - 0.665001542781237 z_k1_s1142
                                 - 0.735001462493912 z_k1_s1143
                                 - 0.595001623068562 z_k1_s1144
                                 - 0.248001032988235 z_k1_s1145
                                 - 0.53200123422499 z_k1_s1146
                                 - 0.667002726178596 z_k1_s1147
                                 - 0.960001467628533 z_k1_s1150
                                 - 0.908001614031695 z_k1_s1151
                                 - 0.767995752509854 z_k1_s1152
                                 - 1.09800133975057 z_k1_s1153
                                 - 0.729000945055637 z_k1_s1154
                                 - 0.928000515108753 z_k1_s1155
                                 - 0.456999625628724 z_k1_s1156
                                 - 0.928996732817438 z_k1_s1157
                                 - 0.555000205449074 z_k1_s1160
                                 - 1.65700392321052 z_k1_s1161
                                 - 0.451000160869067 z_k1_s1162
                                 - 1.70400358636492 z_k1_s1163
                                 - 0.797000374091019 z_k1_s1164
                                 - 0.889998109732601 z_k1_s1165
                                 - 0.866996120188813 z_k1_s1166
                                 - 1.51700061781414 z_k1_s1167
                                 - 0.424999333672919 z_k1_s1170
                                 - 0.296999770918181 z_k1_s1171
                                 - 0.327999553776894 z_k1_s1172
                                 - 0.325999553866673 z_k1_s1173
                                 - 0.674998890951173 z_k1_s1174
                                 - 0.406999334480925 z_k1_s1175
                                 - 0.466998900288131 z_k1_s1176
                                 - 1.38999778010678 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.861192 w_k1_ADM_s000
                                 + 0.782031 w_k1_AXP_s000
                                 + 0.702929 w_k1_DIS_s000
                                 + 0.807239 w_k1_FDX_s000
                                 + 1.01372 w_k1_IBM_s000
                                 + 0.957421 w_k1_INTC_s000
                                 + 0.853768 w_k1_KO_s000
                                 + 0.889738 w_k1_MAT_s000
                                 + 0.882074 w_k1_MSFT_s000
                                 + 0.918863 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.683763 w_k1_ADM_s000
                                 + 0.65585 w_k1_AXP_s000
                                 + 0.544834 w_k1_DIS_s000
                                 + 0.748005 w_k1_FDX_s000
                                 + 0.616362 w_k1_IBM_s000
                                 + 0.831625 w_k1_INTC_s000
                                 + 0.567647 w_k1_KO_s000
                                 + 0.709821 w_k1_MAT_s000
                                 + 0.854992 w_k1_MSFT_s000
                                 + 0.746468 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.32848 w_k1_ADM_s000
                                 + 1.61971 w_k1_AXP_s000
                                 + 1.21657 w_k1_DIS_s000
                                 + 1.22335 w_k1_FDX_s000
                                 + 1.56391 w_k1_IBM_s000
                                 + 1.36545 w_k1_INTC_s000
                                 + 1.32593 w_k1_KO_s000 + 2.21788 w_k1_MAT_s000
                                 + 1.31181 w_k1_MSFT_s000
                                 + 1.19692 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.897586 w_k1_ADM_s000
                                 + 0.723883 w_k1_AXP_s000
                                 + 1.01287 w_k1_DIS_s000
                                 + 0.964374 w_k1_FDX_s000
                                 + 0.801542 w_k1_IBM_s000
                                 + 0.953897 w_k1_INTC_s000
                                 + 0.778404 w_k1_KO_s000
                                 + 0.798039 w_k1_MAT_s000
                                 + 1.11397 w_k1_MSFT_s000
                                 + 0.971813 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.860915 w_k1_ADM_s000
                                 + 1.09361 w_k1_AXP_s000
                                 + 1.20001 w_k1_DIS_s000
                                 + 0.973754 w_k1_FDX_s000
                                 + 1.38401 w_k1_IBM_s000
                                 + 0.958256 w_k1_INTC_s000
                                 + 1.21108 w_k1_KO_s000 + 1.62732 w_k1_MAT_s000
                                 + 1.18595 w_k1_MSFT_s000 + 1.1032 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.83404 w_k1_ADM_s000
                                 + 1.01413 w_k1_AXP_s000
                                 + 0.683882 w_k1_DIS_s000
                                 + 0.902697 w_k1_FDX_s000
                                 + 0.861695 w_k1_IBM_s000
                                 + 0.960708 w_k1_INTC_s000
                                 + 0.79968 w_k1_KO_s000 + 1.18187 w_k1_MAT_s000
                                 + 0.864406 w_k1_MSFT_s000
                                 + 0.977026 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.992672 w_k1_ADM_s000
                                 + 0.944806 w_k1_AXP_s000
                                 + 0.889964 w_k1_DIS_s000
                                 + 0.89099 w_k1_FDX_s000 + 1.1289 w_k1_IBM_s000
                                 + 0.9055 w_k1_INTC_s000 + 1.05058 w_k1_KO_s000
                                 + 1.53868 w_k1_MAT_s000
                                 + 0.946342 w_k1_MSFT_s000
                                 + 1.04506 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.22441 w_k1_ADM_s000
                                 + 1.30605 w_k1_AXP_s000
                                 + 1.34196 w_k1_DIS_s000
                                 + 1.29692 w_k1_FDX_s000
                                 + 0.922011 w_k1_IBM_s000
                                 + 0.95561 w_k1_INTC_s000
                                 + 1.03626 w_k1_KO_s000 + 1.72542 w_k1_MAT_s000
                                 + 1.14828 w_k1_MSFT_s000
                                 + 1.16691 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.48595 w_k1_ADM_s001
                                 + 1.55343 w_k1_AXP_s001
                                 + 1.67176 w_k1_DIS_s001
                                 + 1.56508 w_k1_FDX_s001
                                 + 0.857149 w_k1_IBM_s001
                                 + 1.29837 w_k1_INTC_s001 + 1.1625 w_k1_KO_s001
                                 + 1.15852 w_k1_MAT_s001
                                 + 1.39846 w_k1_MSFT_s001
                                 + 1.23422 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.643751 w_k1_ADM_s001
                                 + 0.609663 w_k1_AXP_s001
                                 + 0.757955 w_k1_DIS_s001
                                 + 0.667525 w_k1_FDX_s001
                                 + 0.989257 w_k1_IBM_s001
                                 + 0.861198 w_k1_INTC_s001
                                 + 0.789758 w_k1_KO_s001
                                 + 0.613328 w_k1_MAT_s001
                                 + 0.861178 w_k1_MSFT_s001
                                 + 0.861929 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.12068 w_k1_ADM_s001
                                 + 1.16642 w_k1_AXP_s001
                                 + 1.19675 w_k1_DIS_s001
                                 + 1.12777 w_k1_FDX_s001
                                 + 0.679738 w_k1_IBM_s001
                                 + 1.13702 w_k1_INTC_s001
                                 + 0.82206 w_k1_KO_s001
                                 + 0.543732 w_k1_MAT_s001
                                 + 1.15406 w_k1_MSFT_s001
                                 + 1.03897 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.16195 w_k1_ADM_s001
                                 + 1.20674 w_k1_AXP_s001
                                 + 1.76535 w_k1_DIS_s001
                                 + 1.12367 w_k1_FDX_s001 + 1.617 w_k1_IBM_s001
                                 + 1.06613 w_k1_INTC_s001
                                 + 1.54605 w_k1_KO_s001 + 1.49391 w_k1_MAT_s001
                                 + 1.16849 w_k1_MSFT_s001
                                 + 1.30444 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.08396 w_k1_ADM_s001
                                 + 0.913502 w_k1_AXP_s001
                                 + 1.16661 w_k1_DIS_s001
                                 + 1.06832 w_k1_FDX_s001
                                 + 0.839837 w_k1_IBM_s001
                                 + 0.923553 w_k1_INTC_s001
                                 + 0.931022 w_k1_KO_s001
                                 + 0.702918 w_k1_MAT_s001
                                 + 1.05347 w_k1_MSFT_s001
                                 + 1.09098 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.826972 w_k1_ADM_s001
                                 + 0.97788 w_k1_AXP_s001
                                 + 1.04317 w_k1_DIS_s001
                                 + 0.817198 w_k1_FDX_s001
                                 + 1.25801 w_k1_IBM_s001
                                 + 1.12691 w_k1_INTC_s001
                                 + 1.16296 w_k1_KO_s001 + 1.2944 w_k1_MAT_s001
                                 + 1.13957 w_k1_MSFT_s001
                                 + 0.985694 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.11211 w_k1_ADM_s001
                                 + 1.29202 w_k1_AXP_s001
                                 + 1.46682 w_k1_DIS_s001
                                 + 1.19075 w_k1_FDX_s001
                                 + 1.10921 w_k1_IBM_s001
                                 + 0.973798 w_k1_INTC_s001
                                 + 1.14614 w_k1_KO_s001 + 1.17467 w_k1_MAT_s001
                                 + 1.1312 w_k1_MSFT_s001 + 1.07685 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.951462 w_k1_ADM_s001
                                 + 0.621909 w_k1_AXP_s001
                                 + 0.706066 w_k1_DIS_s001
                                 + 0.79069 w_k1_FDX_s001
                                 + 0.693094 w_k1_IBM_s001
                                 + 0.868541 w_k1_INTC_s001
                                 + 0.839052 w_k1_KO_s001
                                 + 0.606047 w_k1_MAT_s001
                                 + 0.864515 w_k1_MSFT_s001
                                 + 0.855804 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.38156 w_k1_ADM_s002
                                 + 1.50125 w_k1_AXP_s002
                                 + 1.56428 w_k1_DIS_s002
                                 + 1.42063 w_k1_FDX_s002 + 1.0383 w_k1_IBM_s002
                                 + 1.26708 w_k1_INTC_s002
                                 + 1.19613 w_k1_KO_s002 + 1.31378 w_k1_MAT_s002
                                 + 1.36871 w_k1_MSFT_s002
                                 + 1.23336 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.802559 w_k1_ADM_s002
                                 + 1.031 w_k1_AXP_s002 + 0.95577 w_k1_DIS_s002
                                 + 0.983978 w_k1_FDX_s002
                                 + 1.00337 w_k1_IBM_s002
                                 + 1.01177 w_k1_INTC_s002 + 1.0389 w_k1_KO_s002
                                 + 1.12392 w_k1_MAT_s002
                                 + 0.98519 w_k1_MSFT_s002
                                 + 0.913782 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.11159 w_k1_ADM_s002
                                 + 1.14985 w_k1_AXP_s002
                                 + 1.23979 w_k1_DIS_s002 + 1.1268 w_k1_FDX_s002
                                 + 0.955636 w_k1_IBM_s002
                                 + 1.06318 w_k1_INTC_s002
                                 + 1.02046 w_k1_KO_s002
                                 + 0.988822 w_k1_MAT_s002
                                 + 1.13362 w_k1_MSFT_s002
                                 + 1.08604 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.26778 w_k1_ADM_s002
                                 + 1.38766 w_k1_AXP_s002
                                 + 1.23787 w_k1_DIS_s002
                                 + 1.06694 w_k1_FDX_s002
                                 + 1.41138 w_k1_IBM_s002
                                 + 1.09672 w_k1_INTC_s002
                                 + 1.24408 w_k1_KO_s002 + 1.84902 w_k1_MAT_s002
                                 + 1.13099 w_k1_MSFT_s002
                                 + 1.15584 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.55652 w_k1_ADM_s002
                                 + 1.46923 w_k1_AXP_s002
                                 + 2.00381 w_k1_DIS_s002 + 1.4948 w_k1_FDX_s002
                                 + 1.35602 w_k1_IBM_s002
                                 + 1.23209 w_k1_INTC_s002
                                 + 1.29314 w_k1_KO_s002 + 1.48099 w_k1_MAT_s002
                                 + 1.51593 w_k1_MSFT_s002
                                 + 1.20224 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.887962 w_k1_ADM_s002
                                 + 0.78224 w_k1_AXP_s002
                                 + 0.76879 w_k1_DIS_s002
                                 + 0.856932 w_k1_FDX_s002
                                 + 0.646638 w_k1_IBM_s002
                                 + 0.918579 w_k1_INTC_s002
                                 + 0.819233 w_k1_KO_s002
                                 + 0.70243 w_k1_MAT_s002
                                 + 0.938367 w_k1_MSFT_s002
                                 + 0.802155 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.782644 w_k1_ADM_s002
                                 + 0.715255 w_k1_AXP_s002
                                 + 0.569716 w_k1_DIS_s002
                                 + 0.687768 w_k1_FDX_s002
                                 + 0.812184 w_k1_IBM_s002
                                 + 0.58605 w_k1_INTC_s002
                                 + 0.844063 w_k1_KO_s002
                                 + 1.08373 w_k1_MAT_s002
                                 + 0.781068 w_k1_MSFT_s002
                                 + 0.791672 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.913438 w_k1_ADM_s002
                                 + 0.935079 w_k1_AXP_s002
                                 + 0.95199 w_k1_DIS_s002
                                 + 0.940225 w_k1_FDX_s002
                                 + 0.75925 w_k1_IBM_s002
                                 + 0.786951 w_k1_INTC_s002
                                 + 0.788174 w_k1_KO_s002
                                 + 0.875703 w_k1_MAT_s002
                                 + 0.909703 w_k1_MSFT_s002
                                 + 0.901753 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.42929 w_k1_ADM_s003
                                 + 1.3022 w_k1_AXP_s003 + 1.56641 w_k1_DIS_s003
                                 + 1.33709 w_k1_FDX_s003
                                 + 0.969043 w_k1_IBM_s003
                                 + 1.35776 w_k1_INTC_s003
                                 + 1.36057 w_k1_KO_s003 + 1.15311 w_k1_MAT_s003
                                 + 1.47123 w_k1_MSFT_s003
                                 + 1.30592 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.08128 w_k1_ADM_s003
                                 + 0.835025 w_k1_AXP_s003
                                 + 1.01223 w_k1_DIS_s003
                                 + 0.904534 w_k1_FDX_s003
                                 + 1.27984 w_k1_IBM_s003
                                 + 0.993534 w_k1_INTC_s003
                                 + 1.2812 w_k1_KO_s003 + 0.929852 w_k1_MAT_s003
                                 + 1.10038 w_k1_MSFT_s003
                                 + 0.921276 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.08915 w_k1_ADM_s003
                                 + 1.33071 w_k1_AXP_s003
                                 + 1.43737 w_k1_DIS_s003 + 1.1711 w_k1_FDX_s003
                                 + 1.59583 w_k1_IBM_s003
                                 + 1.31904 w_k1_INTC_s003
                                 + 1.36902 w_k1_KO_s003 + 1.44892 w_k1_MAT_s003
                                 + 1.3628 w_k1_MSFT_s003 + 1.1397 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.28126 w_k1_ADM_s003
                                 + 1.01277 w_k1_AXP_s003
                                 + 1.00314 w_k1_DIS_s003
                                 + 1.21284 w_k1_FDX_s003
                                 + 0.818339 w_k1_IBM_s003
                                 + 1.29817 w_k1_INTC_s003
                                 + 0.970702 w_k1_KO_s003
                                 + 0.711781 w_k1_MAT_s003
                                 + 1.19984 w_k1_MSFT_s003
                                 + 1.02886 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.07476 w_k1_ADM_s003
                                 + 0.987113 w_k1_AXP_s003
                                 + 1.09094 w_k1_DIS_s003
                                 + 1.00163 w_k1_FDX_s003
                                 + 1.07423 w_k1_IBM_s003
                                 + 1.05745 w_k1_INTC_s003
                                 + 1.09921 w_k1_KO_s003 + 1.11444 w_k1_MAT_s003
                                 + 1.14923 w_k1_MSFT_s003
                                 + 1.03538 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.25085 w_k1_ADM_s003
                                 + 1.21453 w_k1_AXP_s003
                                 + 0.861008 w_k1_DIS_s003
                                 + 0.972433 w_k1_FDX_s003
                                 + 0.866742 w_k1_IBM_s003
                                 + 0.967991 w_k1_INTC_s003
                                 + 0.940139 w_k1_KO_s003
                                 + 0.933037 w_k1_MAT_s003
                                 + 0.894506 w_k1_MSFT_s003
                                 + 0.977427 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.01925 w_k1_ADM_s003
                                 + 1.07213 w_k1_AXP_s003
                                 + 1.41111 w_k1_DIS_s003
                                 + 1.17238 w_k1_FDX_s003
                                 + 0.867583 w_k1_IBM_s003
                                 + 1.2174 w_k1_INTC_s003 + 1.14352 w_k1_KO_s003
                                 + 0.925699 w_k1_MAT_s003
                                 + 1.23632 w_k1_MSFT_s003
                                 + 0.95284 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.859344 w_k1_ADM_s003
                                 + 0.702123 w_k1_AXP_s003
                                 + 0.748844 w_k1_DIS_s003
                                 + 0.808134 w_k1_FDX_s003
                                 + 0.738824 w_k1_IBM_s003
                                 + 0.991194 w_k1_INTC_s003
                                 + 0.807594 w_k1_KO_s003
                                 + 0.612284 w_k1_MAT_s003
                                 + 0.909612 w_k1_MSFT_s003
                                 + 0.799424 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.01626 w_k1_ADM_s004
                                 + 1.49157 w_k1_AXP_s004
                                 + 1.72486 w_k1_DIS_s004
                                 + 1.49799 w_k1_FDX_s004
                                 + 1.17889 w_k1_IBM_s004
                                 + 1.22999 w_k1_INTC_s004
                                 + 1.25925 w_k1_KO_s004 + 1.75983 w_k1_MAT_s004
                                 + 1.30336 w_k1_MSFT_s004
                                 + 1.35163 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.16289 w_k1_ADM_s004
                                 + 1.24417 w_k1_AXP_s004
                                 + 1.27641 w_k1_DIS_s004
                                 + 1.26909 w_k1_FDX_s004
                                 + 0.849931 w_k1_IBM_s004
                                 + 1.18806 w_k1_INTC_s004
                                 + 1.05227 w_k1_KO_s004 + 1.12315 w_k1_MAT_s004
                                 + 1.22845 w_k1_MSFT_s004
                                 + 1.23233 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.992975 w_k1_ADM_s004
                                 + 0.839366 w_k1_AXP_s004
                                 + 0.650797 w_k1_DIS_s004
                                 + 0.836975 w_k1_FDX_s004
                                 + 0.815225 w_k1_IBM_s004
                                 + 0.683319 w_k1_INTC_s004
                                 + 0.850965 w_k1_KO_s004
                                 + 1.18278 w_k1_MAT_s004
                                 + 0.832192 w_k1_MSFT_s004
                                 + 1.0692 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.06757 w_k1_ADM_s004
                                 + 0.65361 w_k1_AXP_s004
                                 + 0.753553 w_k1_DIS_s004
                                 + 0.745528 w_k1_FDX_s004
                                 + 0.968752 w_k1_IBM_s004
                                 + 0.71027 w_k1_INTC_s004
                                 + 1.07982 w_k1_KO_s004
                                 + 0.969046 w_k1_MAT_s004
                                 + 0.812022 w_k1_MSFT_s004
                                 + 1.10721 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.18645 w_k1_ADM_s004
                                 + 1.35623 w_k1_AXP_s004
                                 + 1.00977 w_k1_DIS_s004
                                 + 1.01345 w_k1_FDX_s004
                                 + 1.47205 w_k1_IBM_s004
                                 + 1.05559 w_k1_INTC_s004
                                 + 1.06228 w_k1_KO_s004 + 1.56658 w_k1_MAT_s004
                                 + 1.07673 w_k1_MSFT_s004
                                 + 1.26528 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.02251 w_k1_ADM_s004
                                 + 1.0035 w_k1_AXP_s004
                                 + 0.948561 w_k1_DIS_s004
                                 + 1.04055 w_k1_FDX_s004
                                 + 1.01155 w_k1_IBM_s004
                                 + 0.927036 w_k1_INTC_s004
                                 + 1.04605 w_k1_KO_s004 + 1.1568 w_k1_MAT_s004
                                 + 1.17391 w_k1_MSFT_s004
                                 + 1.21038 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.87602 w_k1_ADM_s004
                                 + 0.667317 w_k1_AXP_s004
                                 + 0.775953 w_k1_DIS_s004
                                 + 0.848302 w_k1_FDX_s004
                                 + 0.645937 w_k1_IBM_s004
                                 + 0.691406 w_k1_INTC_s004
                                 + 0.764868 w_k1_KO_s004
                                 + 0.642617 w_k1_MAT_s004
                                 + 0.930744 w_k1_MSFT_s004
                                 + 0.918815 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.05978 w_k1_ADM_s004
                                 + 1.69656 w_k1_AXP_s004
                                 + 1.64857 w_k1_DIS_s004
                                 + 1.18452 w_k1_FDX_s004
                                 + 1.81905 w_k1_IBM_s004
                                 + 1.15437 w_k1_INTC_s004
                                 + 1.76775 w_k1_KO_s004 + 2.52958 w_k1_MAT_s004
                                 + 1.2351 w_k1_MSFT_s004 + 1.4218 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.936249 w_k1_ADM_s005
                                 + 0.969659 w_k1_AXP_s005
                                 + 1.01632 w_k1_DIS_s005
                                 + 1.04761 w_k1_FDX_s005
                                 + 0.941731 w_k1_IBM_s005
                                 + 1.06581 w_k1_INTC_s005
                                 + 0.918987 w_k1_KO_s005
                                 + 0.966739 w_k1_MAT_s005
                                 + 0.999901 w_k1_MSFT_s005
                                 + 1.01178 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.26737 w_k1_ADM_s005
                                 + 0.992045 w_k1_AXP_s005
                                 + 1.12506 w_k1_DIS_s005 + 1.0439 w_k1_FDX_s005
                                 + 1.15981 w_k1_IBM_s005
                                 + 0.789032 w_k1_INTC_s005
                                 + 1.08227 w_k1_KO_s005 + 1.03794 w_k1_MAT_s005
                                 + 0.985307 w_k1_MSFT_s005
                                 + 1.12588 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.668424 w_k1_ADM_s005
                                 + 0.549694 w_k1_AXP_s005
                                 + 0.681931 w_k1_DIS_s005
                                 + 0.787493 w_k1_FDX_s005
                                 + 0.668826 w_k1_IBM_s005
                                 + 0.80725 w_k1_INTC_s005
                                 + 0.634136 w_k1_KO_s005
                                 + 0.495283 w_k1_MAT_s005
                                 + 0.794059 w_k1_MSFT_s005
                                 + 0.76668 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.00749 w_k1_ADM_s005
                                 + 1.08642 w_k1_AXP_s005 + 1.019 w_k1_DIS_s005
                                 + 1.0654 w_k1_FDX_s005
                                 + 0.744414 w_k1_IBM_s005
                                 + 0.899607 w_k1_INTC_s005
                                 + 0.807785 w_k1_KO_s005
                                 + 0.930417 w_k1_MAT_s005
                                 + 1.08193 w_k1_MSFT_s005
                                 + 1.01078 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.806657 w_k1_ADM_s005
                                 + 0.802375 w_k1_AXP_s005
                                 + 0.79631 w_k1_DIS_s005
                                 + 0.812531 w_k1_FDX_s005
                                 + 1.03291 w_k1_IBM_s005
                                 + 0.781653 w_k1_INTC_s005
                                 + 0.83802 w_k1_KO_s005
                                 + 0.878872 w_k1_MAT_s005
                                 + 0.78921 w_k1_MSFT_s005
                                 + 0.908497 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.79499 w_k1_ADM_s005
                                 + 1.3893 w_k1_AXP_s005 + 1.62786 w_k1_DIS_s005
                                 + 1.38126 w_k1_FDX_s005
                                 + 1.26385 w_k1_IBM_s005
                                 + 1.10437 w_k1_INTC_s005
                                 + 1.24372 w_k1_KO_s005 + 1.03647 w_k1_MAT_s005
                                 + 1.35879 w_k1_MSFT_s005
                                 + 1.26008 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11481 w_k1_ADM_s005
                                 + 1.11708 w_k1_AXP_s005
                                 + 1.19011 w_k1_DIS_s005
                                 + 1.15885 w_k1_FDX_s005
                                 + 1.79755 w_k1_IBM_s005
                                 + 1.11108 w_k1_INTC_s005
                                 + 1.41688 w_k1_KO_s005 + 1.76484 w_k1_MAT_s005
                                 + 1.12567 w_k1_MSFT_s005
                                 + 1.36824 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.664189 w_k1_ADM_s005
                                 + 0.879234 w_k1_AXP_s005
                                 + 1.04055 w_k1_DIS_s005
                                 + 0.86038 w_k1_FDX_s005
                                 + 1.44453 w_k1_IBM_s005
                                 + 0.820449 w_k1_INTC_s005
                                 + 1.04187 w_k1_KO_s005 + 1.27496 w_k1_MAT_s005
                                 + 0.87345 w_k1_MSFT_s005
                                 + 1.05399 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.855185 w_k1_ADM_s006
                                 + 0.999843 w_k1_AXP_s006
                                 + 0.775567 w_k1_DIS_s006
                                 + 0.871281 w_k1_FDX_s006
                                 + 0.9938 w_k1_IBM_s006
                                 + 0.817666 w_k1_INTC_s006
                                 + 0.706365 w_k1_KO_s006
                                 + 0.838218 w_k1_MAT_s006
                                 + 0.721501 w_k1_MSFT_s006
                                 + 0.87655 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.960505 w_k1_ADM_s006
                                 + 1.17572 w_k1_AXP_s006
                                 + 1.06568 w_k1_DIS_s006
                                 + 1.16019 w_k1_FDX_s006
                                 + 0.714711 w_k1_IBM_s006
                                 + 0.872093 w_k1_INTC_s006
                                 + 0.762231 w_k1_KO_s006
                                 + 0.932242 w_k1_MAT_s006
                                 + 0.929562 w_k1_MSFT_s006
                                 + 0.99106 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.755862 w_k1_ADM_s006
                                 + 0.653154 w_k1_AXP_s006
                                 + 0.66557 w_k1_DIS_s006
                                 + 0.763685 w_k1_FDX_s006
                                 + 0.814337 w_k1_IBM_s006
                                 + 0.706799 w_k1_INTC_s006
                                 + 0.758425 w_k1_KO_s006
                                 + 0.585177 w_k1_MAT_s006
                                 + 0.717338 w_k1_MSFT_s006
                                 + 0.87931 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.581601 w_k1_ADM_s006
                                 + 0.822273 w_k1_AXP_s006
                                 + 0.564033 w_k1_DIS_s006
                                 + 0.737345 w_k1_FDX_s006
                                 + 0.592214 w_k1_IBM_s006
                                 + 0.866221 w_k1_INTC_s006
                                 + 0.590492 w_k1_KO_s006
                                 + 0.757972 w_k1_MAT_s006
                                 + 0.803136 w_k1_MSFT_s006
                                 + 0.757043 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.954179 w_k1_ADM_s006
                                 + 0.878743 w_k1_AXP_s006
                                 + 0.914093 w_k1_DIS_s006
                                 + 0.949742 w_k1_FDX_s006
                                 + 0.903817 w_k1_IBM_s006
                                 + 1.01297 w_k1_INTC_s006
                                 + 0.848883 w_k1_KO_s006
                                 + 1.18411 w_k1_MAT_s006
                                 + 0.949215 w_k1_MSFT_s006
                                 + 0.929394 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.921667 w_k1_ADM_s006
                                 + 1.22711 w_k1_AXP_s006
                                 + 1.10241 w_k1_DIS_s006
                                 + 1.12898 w_k1_FDX_s006
                                 + 1.20807 w_k1_IBM_s006
                                 + 0.992699 w_k1_INTC_s006
                                 + 0.98998 w_k1_KO_s006 + 1.67184 w_k1_MAT_s006
                                 + 1.03104 w_k1_MSFT_s006
                                 + 0.958835 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.822359 w_k1_ADM_s006
                                 + 1.0309 w_k1_AXP_s006 + 1.15642 w_k1_DIS_s006
                                 + 1.00979 w_k1_FDX_s006
                                 + 1.39346 w_k1_IBM_s006
                                 + 0.91905 w_k1_INTC_s006
                                 + 1.00849 w_k1_KO_s006 + 1.74367 w_k1_MAT_s006
                                 + 1.03546 w_k1_MSFT_s006
                                 + 1.03648 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.31889 w_k1_ADM_s006
                                 + 1.28258 w_k1_AXP_s006
                                 + 1.34047 w_k1_DIS_s006
                                 + 1.29641 w_k1_FDX_s006
                                 + 1.01339 w_k1_IBM_s006
                                 + 1.00864 w_k1_INTC_s006
                                 + 0.950131 w_k1_KO_s006
                                 + 1.02382 w_k1_MAT_s006
                                 + 1.15425 w_k1_MSFT_s006
                                 + 1.16066 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.909285 w_k1_ADM_s007
                                 + 1.08105 w_k1_AXP_s007
                                 + 1.21853 w_k1_DIS_s007
                                 + 1.02371 w_k1_FDX_s007
                                 + 1.55045 w_k1_IBM_s007
                                 + 1.15411 w_k1_INTC_s007
                                 + 1.41354 w_k1_KO_s007 + 1.79033 w_k1_MAT_s007
                                 + 0.976701 w_k1_MSFT_s007
                                 + 1.12015 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.29607 w_k1_ADM_s007
                                 + 1.46227 w_k1_AXP_s007
                                 + 1.64638 w_k1_DIS_s007 + 1.3236 w_k1_FDX_s007
                                 + 1.3236 w_k1_IBM_s007
                                 + 1.34806 w_k1_INTC_s007
                                 + 1.43972 w_k1_KO_s007 + 1.26687 w_k1_MAT_s007
                                 + 1.19012 w_k1_MSFT_s007
                                 + 1.31428 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.96014 w_k1_ADM_s007
                                 + 0.817745 w_k1_AXP_s007
                                 + 0.801516 w_k1_DIS_s007
                                 + 0.920414 w_k1_FDX_s007
                                 + 0.867623 w_k1_IBM_s007
                                 + 1.02836 w_k1_INTC_s007
                                 + 0.837694 w_k1_KO_s007
                                 + 0.934537 w_k1_MAT_s007
                                 + 0.854647 w_k1_MSFT_s007
                                 + 0.879894 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.09463 w_k1_ADM_s007
                                 + 1.08463 w_k1_AXP_s007
                                 + 1.11799 w_k1_DIS_s007 + 1.1475 w_k1_FDX_s007
                                 + 0.802308 w_k1_IBM_s007
                                 + 1.46361 w_k1_INTC_s007
                                 + 0.936034 w_k1_KO_s007
                                 + 1.00545 w_k1_MAT_s007
                                 + 1.22929 w_k1_MSFT_s007
                                 + 1.00909 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.654108 w_k1_ADM_s007
                                 + 0.776529 w_k1_AXP_s007
                                 + 0.64073 w_k1_DIS_s007
                                 + 0.845859 w_k1_FDX_s007
                                 + 0.737127 w_k1_IBM_s007
                                 + 0.911401 w_k1_INTC_s007
                                 + 0.737258 w_k1_KO_s007
                                 + 0.791146 w_k1_MAT_s007
                                 + 0.78513 w_k1_MSFT_s007
                                 + 0.794681 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.07225 w_k1_ADM_s007
                                 + 1.18255 w_k1_AXP_s007
                                 + 0.763587 w_k1_DIS_s007
                                 + 0.897528 w_k1_FDX_s007
                                 + 1.08285 w_k1_IBM_s007
                                 + 1.04726 w_k1_INTC_s007
                                 + 0.983306 w_k1_KO_s007 + 1.0207 w_k1_MAT_s007
                                 + 0.851145 w_k1_MSFT_s007
                                 + 1.07183 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.703967 w_k1_ADM_s007
                                 + 0.933653 w_k1_AXP_s007
                                 + 0.875921 w_k1_DIS_s007
                                 + 0.882161 w_k1_FDX_s007
                                 + 1.25324 w_k1_IBM_s007
                                 + 1.16143 w_k1_INTC_s007
                                 + 1.09752 w_k1_KO_s007 + 1.37401 w_k1_MAT_s007
                                 + 0.951853 w_k1_MSFT_s007
                                 + 0.997124 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.08748 w_k1_ADM_s007
                                 + 1.46164 w_k1_AXP_s007
                                 + 1.17202 w_k1_DIS_s007
                                 + 1.12608 w_k1_FDX_s007
                                 + 1.04696 w_k1_IBM_s007
                                 + 1.31865 w_k1_INTC_s007
                                 + 1.14375 w_k1_KO_s007 + 1.33965 w_k1_MAT_s007
                                 + 1.02841 w_k1_MSFT_s007
                                 + 1.14261 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.902616 w_k1_ADM_s010
                                 + 0.966851 w_k1_AXP_s010
                                 + 0.972465 w_k1_DIS_s010
                                 + 0.969484 w_k1_FDX_s010
                                 + 1.10821 w_k1_IBM_s010
                                 + 0.730576 w_k1_INTC_s010
                                 + 1.00306 w_k1_KO_s010 + 1.51039 w_k1_MAT_s010
                                 + 0.851766 w_k1_MSFT_s010
                                 + 1.01924 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.941623 w_k1_ADM_s010
                                 + 0.796212 w_k1_AXP_s010
                                 + 0.925605 w_k1_DIS_s010
                                 + 1.03932 w_k1_FDX_s010
                                 + 0.645602 w_k1_IBM_s010
                                 + 0.862268 w_k1_INTC_s010
                                 + 0.737011 w_k1_KO_s010
                                 + 0.657531 w_k1_MAT_s010
                                 + 0.8669 w_k1_MSFT_s010
                                 + 0.908328 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.821879 w_k1_ADM_s010
                                 + 0.785148 w_k1_AXP_s010
                                 + 0.798766 w_k1_DIS_s010
                                 + 0.844383 w_k1_FDX_s010
                                 + 1.05425 w_k1_IBM_s010
                                 + 0.881583 w_k1_INTC_s010
                                 + 0.983513 w_k1_KO_s010
                                 + 0.796166 w_k1_MAT_s010
                                 + 0.830167 w_k1_MSFT_s010
                                 + 0.923955 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.812541 w_k1_ADM_s010
                                 + 0.763517 w_k1_AXP_s010
                                 + 0.601636 w_k1_DIS_s010
                                 + 0.778443 w_k1_FDX_s010
                                 + 0.702061 w_k1_IBM_s010
                                 + 0.697468 w_k1_INTC_s010
                                 + 0.679736 w_k1_KO_s010
                                 + 0.755221 w_k1_MAT_s010
                                 + 0.699151 w_k1_MSFT_s010
                                 + 0.786599 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.05363 w_k1_ADM_s010
                                 + 1.20402 w_k1_AXP_s010
                                 + 0.821364 w_k1_DIS_s010
                                 + 1.07295 w_k1_FDX_s010
                                 + 0.742174 w_k1_IBM_s010
                                 + 0.908002 w_k1_INTC_s010
                                 + 0.910433 w_k1_KO_s010
                                 + 1.27579 w_k1_MAT_s010
                                 + 0.991972 w_k1_MSFT_s010
                                 + 0.96541 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.14147 w_k1_ADM_s010
                                 + 1.33103 w_k1_AXP_s010
                                 + 1.39949 w_k1_DIS_s010
                                 + 1.09118 w_k1_FDX_s010
                                 + 1.57997 w_k1_IBM_s010
                                 + 0.929541 w_k1_INTC_s010
                                 + 1.35596 w_k1_KO_s010 + 1.82904 w_k1_MAT_s010
                                 + 1.11036 w_k1_MSFT_s010
                                 + 1.15698 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.22855 w_k1_ADM_s010
                                 + 1.25501 w_k1_AXP_s010
                                 + 1.03773 w_k1_DIS_s010
                                 + 1.14994 w_k1_FDX_s010 + 1.0196 w_k1_IBM_s010
                                 + 0.874437 w_k1_INTC_s010
                                 + 0.969508 w_k1_KO_s010
                                 + 1.33691 w_k1_MAT_s010
                                 + 0.917217 w_k1_MSFT_s010
                                 + 0.980944 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.18686 w_k1_ADM_s010
                                 + 1.13101 w_k1_AXP_s010
                                 + 1.45888 w_k1_DIS_s010
                                 + 1.28664 w_k1_FDX_s010
                                 + 1.01689 w_k1_IBM_s010
                                 + 0.944133 w_k1_INTC_s010
                                 + 1.09728 w_k1_KO_s010 + 0.87995 w_k1_MAT_s010
                                 + 1.06346 w_k1_MSFT_s010
                                 + 1.05656 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.08946 w_k1_ADM_s011
                                 + 0.825506 w_k1_AXP_s011
                                 + 1.23687 w_k1_DIS_s011
                                 + 1.14045 w_k1_FDX_s011
                                 + 0.730058 w_k1_IBM_s011
                                 + 0.940415 w_k1_INTC_s011
                                 + 1.0589 w_k1_KO_s011 + 0.83079 w_k1_MAT_s011
                                 + 1.22195 w_k1_MSFT_s011
                                 + 1.01554 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.17682 w_k1_ADM_s011
                                 + 1.54188 w_k1_AXP_s011
                                 + 1.24391 w_k1_DIS_s011
                                 + 1.18835 w_k1_FDX_s011
                                 + 1.14735 w_k1_IBM_s011
                                 + 1.03414 w_k1_INTC_s011
                                 + 1.27401 w_k1_KO_s011 + 2.12066 w_k1_MAT_s011
                                 + 1.08562 w_k1_MSFT_s011
                                 + 1.19696 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.878417 w_k1_ADM_s011
                                 + 0.67459 w_k1_AXP_s011
                                 + 0.785135 w_k1_DIS_s011
                                 + 0.784351 w_k1_FDX_s011
                                 + 0.793199 w_k1_IBM_s011
                                 + 0.667372 w_k1_INTC_s011
                                 + 0.859389 w_k1_KO_s011
                                 + 1.00669 w_k1_MAT_s011
                                 + 0.931397 w_k1_MSFT_s011
                                 + 0.772858 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.03266 w_k1_ADM_s011
                                 + 1.21316 w_k1_AXP_s011 + 1.1558 w_k1_DIS_s011
                                 + 1.1448 w_k1_FDX_s011
                                 + 0.813853 w_k1_IBM_s011
                                 + 1.01841 w_k1_INTC_s011
                                 + 1.05041 w_k1_KO_s011 + 1.1836 w_k1_MAT_s011
                                 + 1.10916 w_k1_MSFT_s011
                                 + 1.05116 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.05096 w_k1_ADM_s011
                                 + 0.758828 w_k1_AXP_s011
                                 + 1.10157 w_k1_DIS_s011
                                 + 0.965906 w_k1_FDX_s011
                                 + 1.03592 w_k1_IBM_s011
                                 + 0.828567 w_k1_INTC_s011
                                 + 1.03406 w_k1_KO_s011 + 1.03383 w_k1_MAT_s011
                                 + 1.02742 w_k1_MSFT_s011
                                 + 0.934161 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.75582 w_k1_ADM_s011
                                 + 1.17628 w_k1_AXP_s011
                                 + 1.36843 w_k1_DIS_s011
                                 + 1.13612 w_k1_FDX_s011
                                 + 1.53992 w_k1_IBM_s011
                                 + 1.06309 w_k1_INTC_s011
                                 + 1.30693 w_k1_KO_s011 + 1.80581 w_k1_MAT_s011
                                 + 1.30167 w_k1_MSFT_s011
                                 + 1.08914 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.777571 w_k1_ADM_s011
                                 + 1.01937 w_k1_AXP_s011
                                 + 1.05718 w_k1_DIS_s011
                                 + 1.01302 w_k1_FDX_s011
                                 + 1.00969 w_k1_IBM_s011
                                 + 0.914554 w_k1_INTC_s011
                                 + 0.985504 w_k1_KO_s011
                                 + 1.04525 w_k1_MAT_s011
                                 + 0.982644 w_k1_MSFT_s011
                                 + 0.979948 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.86711 w_k1_ADM_s011
                                 + 0.929761 w_k1_AXP_s011
                                 + 0.794826 w_k1_DIS_s011
                                 + 0.852004 w_k1_FDX_s011
                                 + 1.19706 w_k1_IBM_s011
                                 + 0.84809 w_k1_INTC_s011
                                 + 1.02711 w_k1_KO_s011 + 1.5553 w_k1_MAT_s011
                                 + 0.923251 w_k1_MSFT_s011
                                 + 1.01526 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.655116 w_k1_ADM_s012
                                 + 0.820216 w_k1_AXP_s012
                                 + 0.681564 w_k1_DIS_s012
                                 + 0.722381 w_k1_FDX_s012
                                 + 1.31209 w_k1_IBM_s012
                                 + 0.695025 w_k1_INTC_s012
                                 + 0.845246 w_k1_KO_s012 + 0.8483 w_k1_MAT_s012
                                 + 0.735183 w_k1_MSFT_s012
                                 + 0.821175 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.03649 w_k1_ADM_s012
                                 + 1.48696 w_k1_AXP_s012
                                 + 1.76819 w_k1_DIS_s012
                                 + 1.28952 w_k1_FDX_s012
                                 + 1.29048 w_k1_IBM_s012
                                 + 1.13942 w_k1_INTC_s012
                                 + 1.18972 w_k1_KO_s012 + 1.41057 w_k1_MAT_s012
                                 + 1.14267 w_k1_MSFT_s012
                                 + 1.33873 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.0506 w_k1_ADM_s012
                                 + 1.27128 w_k1_AXP_s012
                                 + 1.01205 w_k1_DIS_s012
                                 + 1.05827 w_k1_FDX_s012
                                 + 0.974504 w_k1_IBM_s012
                                 + 0.845211 w_k1_INTC_s012
                                 + 1.03545 w_k1_KO_s012
                                 + 0.789496 w_k1_MAT_s012
                                 + 0.85885 w_k1_MSFT_s012 + 1.1048 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.787154 w_k1_ADM_s012
                                 + 0.803477 w_k1_AXP_s012
                                 + 0.788337 w_k1_DIS_s012
                                 + 0.784588 w_k1_FDX_s012
                                 + 1.05317 w_k1_IBM_s012
                                 + 0.812686 w_k1_INTC_s012
                                 + 0.958576 w_k1_KO_s012
                                 + 0.759142 w_k1_MAT_s012
                                 + 0.806397 w_k1_MSFT_s012
                                 + 0.820739 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.894828 w_k1_ADM_s012
                                 + 0.720788 w_k1_AXP_s012
                                 + 1.14128 w_k1_DIS_s012
                                 + 0.948046 w_k1_FDX_s012
                                 + 0.916505 w_k1_IBM_s012
                                 + 0.988602 w_k1_INTC_s012
                                 + 1.02069 w_k1_KO_s012
                                 + 0.587177 w_k1_MAT_s012
                                 + 1.01739 w_k1_MSFT_s012
                                 + 0.980985 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.07011 w_k1_ADM_s012
                                 + 1.07596 w_k1_AXP_s012
                                 + 1.46426 w_k1_DIS_s012
                                 + 1.22818 w_k1_FDX_s012
                                 + 0.94872 w_k1_IBM_s012
                                 + 0.973775 w_k1_INTC_s012
                                 + 1.18683 w_k1_KO_s012
                                 + 0.883895 w_k1_MAT_s012
                                 + 1.12947 w_k1_MSFT_s012
                                 + 1.15191 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.735408 w_k1_ADM_s012
                                 + 1.04151 w_k1_AXP_s012
                                 + 0.641157 w_k1_DIS_s012
                                 + 0.777278 w_k1_FDX_s012
                                 + 0.77904 w_k1_IBM_s012
                                 + 0.807917 w_k1_INTC_s012
                                 + 0.816389 w_k1_KO_s012
                                 + 0.858878 w_k1_MAT_s012
                                 + 0.766414 w_k1_MSFT_s012
                                 + 0.923839 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.812892 w_k1_ADM_s012
                                 + 0.870751 w_k1_AXP_s012
                                 + 0.941271 w_k1_DIS_s012
                                 + 0.911543 w_k1_FDX_s012
                                 + 0.615177 w_k1_IBM_s012
                                 + 0.876876 w_k1_INTC_s012
                                 + 0.845494 w_k1_KO_s012
                                 + 0.570847 w_k1_MAT_s012
                                 + 0.927351 w_k1_MSFT_s012
                                 + 0.86206 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.04732 w_k1_ADM_s013
                                 + 1.75725 w_k1_AXP_s013
                                 + 1.68093 w_k1_DIS_s013
                                 + 1.36494 w_k1_FDX_s013
                                 + 1.70798 w_k1_IBM_s013
                                 + 1.31145 w_k1_INTC_s013
                                 + 1.53135 w_k1_KO_s013 + 1.50864 w_k1_MAT_s013
                                 + 1.30682 w_k1_MSFT_s013 + 1.3768 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.784769 w_k1_ADM_s013
                                 + 0.890081 w_k1_AXP_s013
                                 + 1.29572 w_k1_DIS_s013
                                 + 1.04395 w_k1_FDX_s013
                                 + 1.05739 w_k1_IBM_s013
                                 + 1.16709 w_k1_INTC_s013
                                 + 1.15613 w_k1_KO_s013
                                 + 0.807147 w_k1_MAT_s013
                                 + 1.21232 w_k1_MSFT_s013
                                 + 0.944023 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.843029 w_k1_ADM_s013
                                 + 0.839363 w_k1_AXP_s013
                                 + 0.768913 w_k1_DIS_s013
                                 + 0.876702 w_k1_FDX_s013
                                 + 1.07155 w_k1_IBM_s013
                                 + 1.02456 w_k1_INTC_s013
                                 + 0.868735 w_k1_KO_s013 + 1.1696 w_k1_MAT_s013
                                 + 0.889089 w_k1_MSFT_s013
                                 + 1.00436 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.05113 w_k1_ADM_s013
                                 + 0.997334 w_k1_AXP_s013
                                 + 1.08408 w_k1_DIS_s013
                                 + 1.15868 w_k1_FDX_s013
                                 + 0.780206 w_k1_IBM_s013
                                 + 1.11184 w_k1_INTC_s013 + 1.0556 w_k1_KO_s013
                                 + 0.858217 w_k1_MAT_s013
                                 + 1.02008 w_k1_MSFT_s013
                                 + 0.983823 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.00528 w_k1_ADM_s013
                                 + 0.959548 w_k1_AXP_s013
                                 + 0.93229 w_k1_DIS_s013
                                 + 0.925581 w_k1_FDX_s013
                                 + 1.46147 w_k1_IBM_s013
                                 + 0.936478 w_k1_INTC_s013
                                 + 1.0629 w_k1_KO_s013 + 1.64377 w_k1_MAT_s013
                                 + 1.03617 w_k1_MSFT_s013
                                 + 1.20833 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.88966 w_k1_ADM_s013
                                 + 0.831459 w_k1_AXP_s013
                                 + 0.775526 w_k1_DIS_s013
                                 + 0.85449 w_k1_FDX_s013
                                 + 0.730183 w_k1_IBM_s013
                                 + 0.917629 w_k1_INTC_s013
                                 + 0.741637 w_k1_KO_s013
                                 + 0.520506 w_k1_MAT_s013
                                 + 0.881747 w_k1_MSFT_s013
                                 + 0.88532 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.07858 w_k1_ADM_s013
                                 + 1.26294 w_k1_AXP_s013
                                 + 1.10059 w_k1_DIS_s013
                                 + 1.17537 w_k1_FDX_s013
                                 + 1.23487 w_k1_IBM_s013
                                 + 0.983444 w_k1_INTC_s013
                                 + 1.15825 w_k1_KO_s013 + 1.18486 w_k1_MAT_s013
                                 + 1.06426 w_k1_MSFT_s013
                                 + 1.07293 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.20997 w_k1_ADM_s013
                                 + 1.45028 w_k1_AXP_s013
                                 + 1.60732 w_k1_DIS_s013 + 1.5725 w_k1_FDX_s013
                                 + 0.963396 w_k1_IBM_s013
                                 + 1.47321 w_k1_INTC_s013
                                 + 1.24001 w_k1_KO_s013 + 0.89919 w_k1_MAT_s013
                                 + 1.70601 w_k1_MSFT_s013
                                 + 1.24246 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.894535 w_k1_ADM_s014
                                 + 1.20656 w_k1_AXP_s014
                                 + 1.35827 w_k1_DIS_s014
                                 + 1.22949 w_k1_FDX_s014
                                 + 0.95284 w_k1_IBM_s014
                                 + 1.13155 w_k1_INTC_s014
                                 + 1.14207 w_k1_KO_s014 + 1.15802 w_k1_MAT_s014
                                 + 1.16681 w_k1_MSFT_s014
                                 + 0.992446 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.859234 w_k1_ADM_s014
                                 + 0.984718 w_k1_AXP_s014
                                 + 1.10619 w_k1_DIS_s014
                                 + 0.885362 w_k1_FDX_s014
                                 + 1.08343 w_k1_IBM_s014
                                 + 0.950781 w_k1_INTC_s014 + 1.116 w_k1_KO_s014
                                 + 1.30617 w_k1_MAT_s014
                                 + 1.0074 w_k1_MSFT_s014
                                 + 0.990173 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.26602 w_k1_ADM_s014
                                 + 1.53998 w_k1_AXP_s014
                                 + 2.02141 w_k1_DIS_s014
                                 + 1.43168 w_k1_FDX_s014
                                 + 1.37556 w_k1_IBM_s014
                                 + 1.23535 w_k1_INTC_s014
                                 + 1.41751 w_k1_KO_s014 + 1.68881 w_k1_MAT_s014
                                 + 1.30781 w_k1_MSFT_s014 + 1.3057 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.412722 w_k1_ADM_s014
                                 + 0.68252 w_k1_AXP_s014
                                 + 0.549345 w_k1_DIS_s014
                                 + 0.634841 w_k1_FDX_s014
                                 + 0.852688 w_k1_IBM_s014
                                 + 0.770624 w_k1_INTC_s014
                                 + 0.647561 w_k1_KO_s014
                                 + 0.947276 w_k1_MAT_s014
                                 + 0.570058 w_k1_MSFT_s014
                                 + 0.658669 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.971524 w_k1_ADM_s014
                                 + 1.12848 w_k1_AXP_s014
                                 + 1.04188 w_k1_DIS_s014
                                 + 1.09616 w_k1_FDX_s014
                                 + 0.862346 w_k1_IBM_s014
                                 + 0.889599 w_k1_INTC_s014
                                 + 0.980666 w_k1_KO_s014
                                 + 1.08342 w_k1_MAT_s014
                                 + 0.973668 w_k1_MSFT_s014
                                 + 0.973238 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.782967 w_k1_ADM_s014
                                 + 0.884134 w_k1_AXP_s014
                                 + 0.714077 w_k1_DIS_s014
                                 + 0.840879 w_k1_FDX_s014
                                 + 1.21397 w_k1_IBM_s014
                                 + 0.681221 w_k1_INTC_s014
                                 + 0.927035 w_k1_KO_s014
                                 + 1.32733 w_k1_MAT_s014
                                 + 0.763444 w_k1_MSFT_s014
                                 + 0.888591 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.75084 w_k1_ADM_s014
                                 + 0.893463 w_k1_AXP_s014
                                 + 0.754892 w_k1_DIS_s014
                                 + 0.865076 w_k1_FDX_s014
                                 + 0.838858 w_k1_IBM_s014
                                 + 0.764283 w_k1_INTC_s014
                                 + 0.797512 w_k1_KO_s014
                                 + 0.753345 w_k1_MAT_s014
                                 + 0.722993 w_k1_MSFT_s014
                                 + 0.796779 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.732081 w_k1_ADM_s014
                                 + 1.12417 w_k1_AXP_s014
                                 + 1.17415 w_k1_DIS_s014
                                 + 1.01617 w_k1_FDX_s014
                                 + 1.48691 w_k1_IBM_s014
                                 + 0.970297 w_k1_INTC_s014
                                 + 1.3528 w_k1_KO_s014 + 1.76277 w_k1_MAT_s014
                                 + 0.969544 w_k1_MSFT_s014
                                 + 1.0014 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.29325 w_k1_ADM_s015
                                 + 1.25318 w_k1_AXP_s015
                                 + 1.19155 w_k1_DIS_s015
                                 + 1.18395 w_k1_FDX_s015
                                 + 1.37394 w_k1_IBM_s015
                                 + 0.960853 w_k1_INTC_s015
                                 + 1.15125 w_k1_KO_s015 + 1.29944 w_k1_MAT_s015
                                 + 0.999536 w_k1_MSFT_s015
                                 + 1.10412 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.855628 w_k1_ADM_s015
                                 + 0.82546 w_k1_AXP_s015
                                 + 0.740992 w_k1_DIS_s015
                                 + 0.846833 w_k1_FDX_s015
                                 + 1.05627 w_k1_IBM_s015
                                 + 0.746249 w_k1_INTC_s015
                                 + 0.88111 w_k1_KO_s015
                                 + 0.874575 w_k1_MAT_s015
                                 + 0.764436 w_k1_MSFT_s015
                                 + 0.867624 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.11162 w_k1_ADM_s015
                                 + 1.09121 w_k1_AXP_s015
                                 + 1.14504 w_k1_DIS_s015
                                 + 1.01495 w_k1_FDX_s015
                                 + 0.842833 w_k1_IBM_s015
                                 + 0.896754 w_k1_INTC_s015
                                 + 0.9401 w_k1_KO_s015 + 0.718476 w_k1_MAT_s015
                                 + 0.904569 w_k1_MSFT_s015
                                 + 0.971556 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.901811 w_k1_ADM_s015
                                 + 0.848449 w_k1_AXP_s015
                                 + 0.677112 w_k1_DIS_s015
                                 + 0.802074 w_k1_FDX_s015
                                 + 0.755796 w_k1_IBM_s015
                                 + 0.862599 w_k1_INTC_s015
                                 + 0.660197 w_k1_KO_s015
                                 + 0.567312 w_k1_MAT_s015
                                 + 0.763853 w_k1_MSFT_s015
                                 + 0.846873 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.33605 w_k1_ADM_s015
                                 + 1.42304 w_k1_AXP_s015
                                 + 1.59356 w_k1_DIS_s015
                                 + 1.48627 w_k1_FDX_s015
                                 + 0.80608 w_k1_IBM_s015
                                 + 1.33376 w_k1_INTC_s015
                                 + 0.911259 w_k1_KO_s015
                                 + 0.631362 w_k1_MAT_s015
                                 + 1.3741 w_k1_MSFT_s015 + 1.12239 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.982922 w_k1_ADM_s015
                                 + 0.978325 w_k1_AXP_s015
                                 + 1.06668 w_k1_DIS_s015
                                 + 0.896536 w_k1_FDX_s015
                                 + 1.1428 w_k1_IBM_s015
                                 + 0.810145 w_k1_INTC_s015
                                 + 0.95079 w_k1_KO_s015
                                 + 0.853661 w_k1_MAT_s015
                                 + 0.840924 w_k1_MSFT_s015
                                 + 1.0108 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.07397 w_k1_ADM_s015
                                 + 1.31935 w_k1_AXP_s015 + 1.8241 w_k1_DIS_s015
                                 + 1.37132 w_k1_FDX_s015
                                 + 1.43887 w_k1_IBM_s015
                                 + 1.16653 w_k1_INTC_s015
                                 + 1.42072 w_k1_KO_s015 + 1.77089 w_k1_MAT_s015
                                 + 1.28617 w_k1_MSFT_s015
                                 + 1.25118 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.916214 w_k1_ADM_s015
                                 + 1.12751 w_k1_AXP_s015
                                 + 1.22475 w_k1_DIS_s015
                                 + 1.22611 w_k1_FDX_s015
                                 + 0.802704 w_k1_IBM_s015
                                 + 1.03168 w_k1_INTC_s015
                                 + 0.873305 w_k1_KO_s015
                                 + 0.62352 w_k1_MAT_s015
                                 + 1.02694 w_k1_MSFT_s015
                                 + 0.976737 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.25519 w_k1_ADM_s016
                                 + 1.16199 w_k1_AXP_s016
                                 + 1.19431 w_k1_DIS_s016
                                 + 1.20495 w_k1_FDX_s016
                                 + 0.908329 w_k1_IBM_s016
                                 + 1.32605 w_k1_INTC_s016
                                 + 1.00739 w_k1_KO_s016
                                 + 0.896908 w_k1_MAT_s016
                                 + 1.14623 w_k1_MSFT_s016
                                 + 1.11007 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.796245 w_k1_ADM_s016
                                 + 1.00307 w_k1_AXP_s016
                                 + 0.833067 w_k1_DIS_s016
                                 + 0.826292 w_k1_FDX_s016
                                 + 1.10997 w_k1_IBM_s016
                                 + 0.939839 w_k1_INTC_s016
                                 + 1.02308 w_k1_KO_s016 + 1.12376 w_k1_MAT_s016
                                 + 0.795218 w_k1_MSFT_s016
                                 + 0.990327 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.43247 w_k1_ADM_s016
                                 + 1.44566 w_k1_AXP_s016
                                 + 1.81599 w_k1_DIS_s016
                                 + 1.45315 w_k1_FDX_s016
                                 + 1.09805 w_k1_IBM_s016
                                 + 1.23915 w_k1_INTC_s016 + 1.3634 w_k1_KO_s016
                                 + 1.59157 w_k1_MAT_s016
                                 + 1.51239 w_k1_MSFT_s016
                                 + 1.21354 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.939113 w_k1_ADM_s016
                                 + 0.651694 w_k1_AXP_s016
                                 + 0.93697 w_k1_DIS_s016
                                 + 0.86217 w_k1_FDX_s016
                                 + 0.824585 w_k1_IBM_s016
                                 + 0.923782 w_k1_INTC_s016
                                 + 0.961079 w_k1_KO_s016
                                 + 0.712448 w_k1_MAT_s016
                                 + 0.954431 w_k1_MSFT_s016
                                 + 0.945349 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.19395 w_k1_ADM_s016
                                 + 1.55908 w_k1_AXP_s016
                                 + 1.21483 w_k1_DIS_s016
                                 + 1.21938 w_k1_FDX_s016 + 1.1291 w_k1_IBM_s016
                                 + 1.2434 w_k1_INTC_s016 + 1.21912 w_k1_KO_s016
                                 + 1.58706 w_k1_MAT_s016
                                 + 1.09595 w_k1_MSFT_s016
                                 + 1.30035 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.01119 w_k1_ADM_s016
                                 + 1.15412 w_k1_AXP_s016
                                 + 0.892697 w_k1_DIS_s016
                                 + 1.05836 w_k1_FDX_s016 + 0.9827 w_k1_IBM_s016
                                 + 1.12985 w_k1_INTC_s016
                                 + 0.988337 w_k1_KO_s016
                                 + 1.15044 w_k1_MAT_s016
                                 + 0.842969 w_k1_MSFT_s016
                                 + 0.932901 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.13033 w_k1_ADM_s016
                                 + 0.994603 w_k1_AXP_s016
                                 + 1.29753 w_k1_DIS_s016
                                 + 1.09327 w_k1_FDX_s016
                                 + 1.18031 w_k1_IBM_s016
                                 + 1.13178 w_k1_INTC_s016
                                 + 1.31667 w_k1_KO_s016 + 1.13037 w_k1_MAT_s016
                                 + 1.08706 w_k1_MSFT_s016
                                 + 1.10281 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.48816 w_k1_ADM_s016
                                 + 1.70042 w_k1_AXP_s016
                                 + 1.47392 w_k1_DIS_s016
                                 + 1.23321 w_k1_FDX_s016
                                 + 1.66913 w_k1_IBM_s016
                                 + 1.30922 w_k1_INTC_s016
                                 + 1.59905 w_k1_KO_s016 + 2.37176 w_k1_MAT_s016
                                 + 1.2153 w_k1_MSFT_s016 + 1.31633 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.01181 w_k1_ADM_s017
                                 + 0.728432 w_k1_AXP_s017
                                 + 0.804068 w_k1_DIS_s017
                                 + 0.813684 w_k1_FDX_s017
                                 + 1.03937 w_k1_IBM_s017
                                 + 0.759814 w_k1_INTC_s017
                                 + 0.772252 w_k1_KO_s017
                                 + 0.595792 w_k1_MAT_s017
                                 + 0.865828 w_k1_MSFT_s017
                                 + 0.940878 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.07278 w_k1_ADM_s017
                                 + 1.08015 w_k1_AXP_s017
                                 + 1.00317 w_k1_DIS_s017
                                 + 1.08575 w_k1_FDX_s017
                                 + 0.743075 w_k1_IBM_s017
                                 + 0.744927 w_k1_INTC_s017
                                 + 0.878045 w_k1_KO_s017
                                 + 0.976566 w_k1_MAT_s017
                                 + 1.02054 w_k1_MSFT_s017
                                 + 1.06981 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.917206 w_k1_ADM_s017
                                 + 1.09389 w_k1_AXP_s017
                                 + 1.24978 w_k1_DIS_s017
                                 + 1.01612 w_k1_FDX_s017
                                 + 1.21334 w_k1_IBM_s017
                                 + 0.942486 w_k1_INTC_s017
                                 + 1.22079 w_k1_KO_s017 + 1.32039 w_k1_MAT_s017
                                 + 1.13884 w_k1_MSFT_s017
                                 + 1.17314 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.08614 w_k1_ADM_s017
                                 + 1.27708 w_k1_AXP_s017
                                 + 1.46061 w_k1_DIS_s017
                                 + 1.11952 w_k1_FDX_s017
                                 + 1.54989 w_k1_IBM_s017
                                 + 0.920981 w_k1_INTC_s017
                                 + 1.31368 w_k1_KO_s017 + 1.35521 w_k1_MAT_s017
                                 + 1.10117 w_k1_MSFT_s017
                                 + 1.31495 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.35719 w_k1_ADM_s017
                                 + 1.53812 w_k1_AXP_s017 + 1.6015 w_k1_DIS_s017
                                 + 1.3479 w_k1_FDX_s017 + 1.26288 w_k1_IBM_s017
                                 + 0.909449 w_k1_INTC_s017
                                 + 1.35144 w_k1_KO_s017 + 1.67573 w_k1_MAT_s017
                                 + 1.32258 w_k1_MSFT_s017
                                 + 1.30859 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.13193 w_k1_ADM_s017
                                 + 1.08954 w_k1_AXP_s017
                                 + 0.960171 w_k1_DIS_s017
                                 + 0.98731 w_k1_FDX_s017
                                 + 1.08228 w_k1_IBM_s017
                                 + 0.761131 w_k1_INTC_s017
                                 + 0.931242 w_k1_KO_s017
                                 + 1.27679 w_k1_MAT_s017
                                 + 0.927226 w_k1_MSFT_s017
                                 + 1.08314 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.09493 w_k1_ADM_s017
                                 + 1.32428 w_k1_AXP_s017
                                 + 1.20886 w_k1_DIS_s017
                                 + 1.18308 w_k1_FDX_s017
                                 + 0.993762 w_k1_IBM_s017
                                 + 1.1902 w_k1_INTC_s017
                                 + 0.936862 w_k1_KO_s017
                                 + 0.73631 w_k1_MAT_s017
                                 + 1.08698 w_k1_MSFT_s017
                                 + 1.06596 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.796677 w_k1_ADM_s017
                                 + 0.736924 w_k1_AXP_s017
                                 + 0.633366 w_k1_DIS_s017
                                 + 0.758851 w_k1_FDX_s017
                                 + 0.685243 w_k1_IBM_s017
                                 + 0.77036 w_k1_INTC_s017
                                 + 0.640673 w_k1_KO_s017
                                 + 0.738636 w_k1_MAT_s017
                                 + 0.84624 w_k1_MSFT_s017
                                 + 0.797713 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.722302 w_k1_ADM_s100
                                 + 0.606969 w_k1_AXP_s100
                                 + 0.597333 w_k1_DIS_s100
                                 + 0.682727 w_k1_FDX_s100
                                 + 0.657226 w_k1_IBM_s100
                                 + 0.701406 w_k1_INTC_s100
                                 + 0.699373 w_k1_KO_s100
                                 + 0.658111 w_k1_MAT_s100
                                 + 0.750741 w_k1_MSFT_s100
                                 + 0.723513 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.937299 w_k1_ADM_s100
                                 + 0.838079 w_k1_AXP_s100
                                 + 1.02138 w_k1_DIS_s100 + 0.938 w_k1_FDX_s100
                                 + 1.34777 w_k1_IBM_s100
                                 + 0.917733 w_k1_INTC_s100
                                 + 1.14891 w_k1_KO_s100
                                 + 0.915487 w_k1_MAT_s100
                                 + 0.959241 w_k1_MSFT_s100
                                 + 0.919599 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.982335 w_k1_ADM_s100
                                 + 0.798256 w_k1_AXP_s100
                                 + 0.864932 w_k1_DIS_s100
                                 + 0.897336 w_k1_FDX_s100
                                 + 0.809933 w_k1_IBM_s100
                                 + 1.07545 w_k1_INTC_s100
                                 + 1.09289 w_k1_KO_s100
                                 + 0.564535 w_k1_MAT_s100
                                 + 0.910197 w_k1_MSFT_s100
                                 + 0.887782 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.05242 w_k1_ADM_s100
                                 + 0.925704 w_k1_AXP_s100
                                 + 1.2748 w_k1_DIS_s100 + 1.14253 w_k1_FDX_s100
                                 + 0.879435 w_k1_IBM_s100
                                 + 1.07413 w_k1_INTC_s100
                                 + 1.09812 w_k1_KO_s100
                                 + 0.803077 w_k1_MAT_s100
                                 + 1.14608 w_k1_MSFT_s100
                                 + 0.963509 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.00585 w_k1_ADM_s100
                                 + 0.898495 w_k1_AXP_s100
                                 + 0.842478 w_k1_DIS_s100
                                 + 0.970705 w_k1_FDX_s100
                                 + 0.676211 w_k1_IBM_s100
                                 + 1.01315 w_k1_INTC_s100
                                 + 0.925231 w_k1_KO_s100
                                 + 0.573465 w_k1_MAT_s100
                                 + 0.909643 w_k1_MSFT_s100
                                 + 0.879301 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.873528 w_k1_ADM_s100
                                 + 0.718152 w_k1_AXP_s100
                                 + 0.896963 w_k1_DIS_s100
                                 + 0.83025 w_k1_FDX_s100
                                 + 0.965997 w_k1_IBM_s100
                                 + 1.04684 w_k1_INTC_s100
                                 + 1.07423 w_k1_KO_s100
                                 + 0.890293 w_k1_MAT_s100
                                 + 0.993478 w_k1_MSFT_s100
                                 + 0.913339 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.936465 w_k1_ADM_s100
                                 + 1.18235 w_k1_AXP_s100
                                 + 0.846538 w_k1_DIS_s100
                                 + 0.917654 w_k1_FDX_s100
                                 + 1.09467 w_k1_IBM_s100
                                 + 1.01489 w_k1_INTC_s100
                                 + 1.03988 w_k1_KO_s100 + 1.09355 w_k1_MAT_s100
                                 + 0.840998 w_k1_MSFT_s100
                                 + 0.896321 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.44195 w_k1_ADM_s100
                                 + 1.33014 w_k1_AXP_s100
                                 + 1.37411 w_k1_DIS_s100
                                 + 1.35458 w_k1_FDX_s100
                                 + 1.18833 w_k1_IBM_s100
                                 + 1.38682 w_k1_INTC_s100
                                 + 1.42394 w_k1_KO_s100
                                 + 0.997198 w_k1_MAT_s100
                                 + 1.26872 w_k1_MSFT_s100 + 1.0245 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.839709 w_k1_ADM_s101
                                 + 1.02454 w_k1_AXP_s101
                                 + 1.09157 w_k1_DIS_s101
                                 + 1.07063 w_k1_FDX_s101
                                 + 0.956266 w_k1_IBM_s101
                                 + 1.12234 w_k1_INTC_s101
                                 + 1.01466 w_k1_KO_s101
                                 + 0.893141 w_k1_MAT_s101
                                 + 1.15877 w_k1_MSFT_s101 + 1.0017 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.42841 w_k1_ADM_s101
                                 + 1.51786 w_k1_AXP_s101
                                 + 1.56708 w_k1_DIS_s101 + 1.2213 w_k1_FDX_s101
                                 + 1.41779 w_k1_IBM_s101
                                 + 1.16306 w_k1_INTC_s101
                                 + 1.42598 w_k1_KO_s101 + 1.94847 w_k1_MAT_s101
                                 + 1.15869 w_k1_MSFT_s101
                                 + 1.21877 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.736021 w_k1_ADM_s101
                                 + 0.751226 w_k1_AXP_s101
                                 + 0.81301 w_k1_DIS_s101
                                 + 0.78614 w_k1_FDX_s101
                                 + 1.32445 w_k1_IBM_s101
                                 + 0.966282 w_k1_INTC_s101
                                 + 1.19232 w_k1_KO_s101 + 1.34431 w_k1_MAT_s101
                                 + 0.82024 w_k1_MSFT_s101
                                 + 0.843606 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.707043 w_k1_ADM_s101
                                 + 0.619436 w_k1_AXP_s101
                                 + 0.753226 w_k1_DIS_s101
                                 + 0.753458 w_k1_FDX_s101
                                 + 0.90602 w_k1_IBM_s101
                                 + 0.850661 w_k1_INTC_s101
                                 + 0.853132 w_k1_KO_s101
                                 + 1.05963 w_k1_MAT_s101
                                 + 0.768798 w_k1_MSFT_s101
                                 + 0.870254 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.0941 w_k1_ADM_s101
                                 + 1.15991 w_k1_AXP_s101
                                 + 0.987365 w_k1_DIS_s101
                                 + 1.13053 w_k1_FDX_s101
                                 + 0.805096 w_k1_IBM_s101
                                 + 1.08297 w_k1_INTC_s101
                                 + 0.938151 w_k1_KO_s101
                                 + 1.02622 w_k1_MAT_s101
                                 + 1.07557 w_k1_MSFT_s101
                                 + 0.972927 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.6138 w_k1_ADM_s101
                                 + 0.734102 w_k1_AXP_s101
                                 + 0.719344 w_k1_DIS_s101
                                 + 0.859104 w_k1_FDX_s101
                                 + 0.697792 w_k1_IBM_s101
                                 + 0.895993 w_k1_INTC_s101
                                 + 0.708287 w_k1_KO_s101
                                 + 0.657764 w_k1_MAT_s101
                                 + 0.86593 w_k1_MSFT_s101
                                 + 0.87613 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.0499 w_k1_ADM_s101
                                 + 0.903595 w_k1_AXP_s101
                                 + 0.985739 w_k1_DIS_s101
                                 + 0.976672 w_k1_FDX_s101
                                 + 1.24813 w_k1_IBM_s101
                                 + 1.10358 w_k1_INTC_s101
                                 + 1.20236 w_k1_KO_s101 + 1.03456 w_k1_MAT_s101
                                 + 0.943777 w_k1_MSFT_s101
                                 + 0.963857 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.08421 w_k1_ADM_s101
                                 + 1.54754 w_k1_AXP_s101
                                 + 1.58235 w_k1_DIS_s101
                                 + 1.38625 w_k1_FDX_s101
                                 + 0.900929 w_k1_IBM_s101
                                 + 1.2792 w_k1_INTC_s101 + 1.27601 w_k1_KO_s101
                                 + 1.44903 w_k1_MAT_s101
                                 + 1.3161 w_k1_MSFT_s101 + 1.18019 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.04561 w_k1_ADM_s102
                                 + 0.94084 w_k1_AXP_s102
                                 + 0.766033 w_k1_DIS_s102
                                 + 0.951929 w_k1_FDX_s102
                                 + 0.961242 w_k1_IBM_s102
                                 + 0.97005 w_k1_INTC_s102
                                 + 0.892999 w_k1_KO_s102
                                 + 0.851431 w_k1_MAT_s102
                                 + 0.986614 w_k1_MSFT_s102
                                 + 0.962069 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.969308 w_k1_ADM_s102
                                 + 0.897151 w_k1_AXP_s102
                                 + 0.998591 w_k1_DIS_s102
                                 + 1.06827 w_k1_FDX_s102
                                 + 0.77566 w_k1_IBM_s102
                                 + 1.14024 w_k1_INTC_s102
                                 + 0.996535 w_k1_KO_s102
                                 + 0.659052 w_k1_MAT_s102
                                 + 1.21397 w_k1_MSFT_s102
                                 + 0.96969 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.14749 w_k1_ADM_s102
                                 + 1.25685 w_k1_AXP_s102
                                 + 1.32327 w_k1_DIS_s102
                                 + 1.22631 w_k1_FDX_s102
                                 + 1.05267 w_k1_IBM_s102
                                 + 1.27073 w_k1_INTC_s102
                                 + 1.07825 w_k1_KO_s102 + 1.14476 w_k1_MAT_s102
                                 + 1.15887 w_k1_MSFT_s102 + 1.0534 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.26752 w_k1_ADM_s102
                                 + 1.11966 w_k1_AXP_s102
                                 + 1.42851 w_k1_DIS_s102
                                 + 1.37627 w_k1_FDX_s102
                                 + 0.953178 w_k1_IBM_s102
                                 + 1.3064 w_k1_INTC_s102 + 1.23324 w_k1_KO_s102
                                 + 1.02771 w_k1_MAT_s102
                                 + 1.39244 w_k1_MSFT_s102
                                 + 1.04879 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.21396 w_k1_ADM_s102
                                 + 2.04511 w_k1_AXP_s102
                                 + 1.63113 w_k1_DIS_s102
                                 + 1.45168 w_k1_FDX_s102
                                 + 1.42221 w_k1_IBM_s102
                                 + 1.47107 w_k1_INTC_s102
                                 + 1.32394 w_k1_KO_s102 + 1.43054 w_k1_MAT_s102
                                 + 1.25492 w_k1_MSFT_s102 + 1.2437 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.809328 w_k1_ADM_s102
                                 + 0.888025 w_k1_AXP_s102
                                 + 0.939793 w_k1_DIS_s102
                                 + 0.867527 w_k1_FDX_s102
                                 + 1.30115 w_k1_IBM_s102
                                 + 0.92786 w_k1_INTC_s102
                                 + 1.18303 w_k1_KO_s102 + 1.01169 w_k1_MAT_s102
                                 + 0.937015 w_k1_MSFT_s102
                                 + 0.923068 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.09319 w_k1_ADM_s102
                                 + 1.40504 w_k1_AXP_s102
                                 + 1.18585 w_k1_DIS_s102 + 1.0656 w_k1_FDX_s102
                                 + 1.41229 w_k1_IBM_s102 + 1.357 w_k1_INTC_s102
                                 + 1.28349 w_k1_KO_s102 + 1.45653 w_k1_MAT_s102
                                 + 1.21781 w_k1_MSFT_s102
                                 + 1.07015 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.817442 w_k1_ADM_s102
                                 + 0.62459 w_k1_AXP_s102
                                 + 0.588413 w_k1_DIS_s102
                                 + 0.792998 w_k1_FDX_s102
                                 + 0.711203 w_k1_IBM_s102
                                 + 0.948237 w_k1_INTC_s102
                                 + 0.706794 w_k1_KO_s102
                                 + 0.466084 w_k1_MAT_s102
                                 + 0.82466 w_k1_MSFT_s102
                                 + 0.813846 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.775553 w_k1_ADM_s103
                                 + 0.601696 w_k1_AXP_s103
                                 + 0.716677 w_k1_DIS_s103
                                 + 0.698662 w_k1_FDX_s103
                                 + 0.980841 w_k1_IBM_s103
                                 + 0.925138 w_k1_INTC_s103
                                 + 0.78225 w_k1_KO_s103
                                 + 0.671221 w_k1_MAT_s103
                                 + 0.851427 w_k1_MSFT_s103
                                 + 0.827443 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.973901 w_k1_ADM_s103
                                 + 1.10577 w_k1_AXP_s103
                                 + 0.935263 w_k1_DIS_s103
                                 + 0.988691 w_k1_FDX_s103
                                 + 1.0579 w_k1_IBM_s103
                                 + 0.969565 w_k1_INTC_s103
                                 + 0.804683 w_k1_KO_s103
                                 + 0.91677 w_k1_MAT_s103
                                 + 0.930185 w_k1_MSFT_s103
                                 + 0.914356 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05627 w_k1_ADM_s103
                                 + 1.44729 w_k1_AXP_s103
                                 + 1.34824 w_k1_DIS_s103
                                 + 1.16446 w_k1_FDX_s103
                                 + 1.46068 w_k1_IBM_s103
                                 + 1.19438 w_k1_INTC_s103
                                 + 1.26084 w_k1_KO_s103 + 1.04023 w_k1_MAT_s103
                                 + 1.10535 w_k1_MSFT_s103
                                 + 1.21155 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.53549 w_k1_ADM_s103
                                 + 1.7233 w_k1_AXP_s103 + 1.7229 w_k1_DIS_s103
                                 + 1.59932 w_k1_FDX_s103
                                 + 1.30749 w_k1_IBM_s103
                                 + 1.35755 w_k1_INTC_s103
                                 + 1.59955 w_k1_KO_s103 + 1.66343 w_k1_MAT_s103
                                 + 1.6253 w_k1_MSFT_s103 + 1.53849 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.23824 w_k1_ADM_s103
                                 + 1.06118 w_k1_AXP_s103
                                 + 1.24674 w_k1_DIS_s103
                                 + 1.20031 w_k1_FDX_s103
                                 + 0.885894 w_k1_IBM_s103
                                 + 1.13315 w_k1_INTC_s103
                                 + 1.12055 w_k1_KO_s103
                                 + 0.637987 w_k1_MAT_s103
                                 + 1.12754 w_k1_MSFT_s103
                                 + 1.16312 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.75386 w_k1_ADM_s103
                                 + 0.838935 w_k1_AXP_s103
                                 + 0.852777 w_k1_DIS_s103
                                 + 0.866685 w_k1_FDX_s103
                                 + 0.698624 w_k1_IBM_s103
                                 + 1.08598 w_k1_INTC_s103
                                 + 0.821857 w_k1_KO_s103
                                 + 0.606983 w_k1_MAT_s103
                                 + 0.973354 w_k1_MSFT_s103
                                 + 0.903478 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.758435 w_k1_ADM_s103
                                 + 0.724377 w_k1_AXP_s103
                                 + 0.557099 w_k1_DIS_s103
                                 + 0.779595 w_k1_FDX_s103
                                 + 0.731569 w_k1_IBM_s103
                                 + 0.899387 w_k1_INTC_s103
                                 + 0.680264 w_k1_KO_s103
                                 + 0.620883 w_k1_MAT_s103
                                 + 0.785174 w_k1_MSFT_s103
                                 + 0.771718 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.885898 w_k1_ADM_s103
                                 + 0.858319 w_k1_AXP_s103
                                 + 0.790464 w_k1_DIS_s103
                                 + 0.831951 w_k1_FDX_s103
                                 + 0.999162 w_k1_IBM_s103
                                 + 0.859889 w_k1_INTC_s103
                                 + 0.915667 w_k1_KO_s103
                                 + 1.01668 w_k1_MAT_s103
                                 + 0.871127 w_k1_MSFT_s103
                                 + 0.906968 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.21142 w_k1_ADM_s104
                                 + 1.05259 w_k1_AXP_s104
                                 + 1.04679 w_k1_DIS_s104
                                 + 1.05415 w_k1_FDX_s104
                                 + 0.761241 w_k1_IBM_s104
                                 + 1.08078 w_k1_INTC_s104 + 1.1468 w_k1_KO_s104
                                 + 0.848915 w_k1_MAT_s104
                                 + 1.17489 w_k1_MSFT_s104
                                 + 0.967165 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.1072 w_k1_ADM_s104
                                 + 1.74389 w_k1_AXP_s104
                                 + 1.48586 w_k1_DIS_s104
                                 + 1.38513 w_k1_FDX_s104
                                 + 1.27108 w_k1_IBM_s104
                                 + 1.34517 w_k1_INTC_s104
                                 + 1.22751 w_k1_KO_s104 + 1.434 w_k1_MAT_s104
                                 + 1.31377 w_k1_MSFT_s104
                                 + 1.04476 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.80843 w_k1_ADM_s104
                                 + 1.32151 w_k1_AXP_s104
                                 + 1.14532 w_k1_DIS_s104
                                 + 0.978315 w_k1_FDX_s104
                                 + 1.53794 w_k1_IBM_s104
                                 + 1.51801 w_k1_INTC_s104
                                 + 1.41497 w_k1_KO_s104 + 1.19096 w_k1_MAT_s104
                                 + 1.17075 w_k1_MSFT_s104
                                 + 0.827811 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.824346 w_k1_ADM_s104
                                 + 0.860131 w_k1_AXP_s104
                                 + 0.789869 w_k1_DIS_s104
                                 + 0.813613 w_k1_FDX_s104
                                 + 1.04598 w_k1_IBM_s104
                                 + 1.13903 w_k1_INTC_s104
                                 + 0.961574 w_k1_KO_s104
                                 + 0.807142 w_k1_MAT_s104
                                 + 0.936933 w_k1_MSFT_s104
                                 + 0.882768 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.634886 w_k1_ADM_s104
                                 + 0.61382 w_k1_AXP_s104
                                 + 0.597554 w_k1_DIS_s104
                                 + 0.752686 w_k1_FDX_s104
                                 + 0.71184 w_k1_IBM_s104
                                 + 0.871801 w_k1_INTC_s104
                                 + 0.770473 w_k1_KO_s104
                                 + 0.697632 w_k1_MAT_s104
                                 + 0.815694 w_k1_MSFT_s104
                                 + 0.744858 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.881977 w_k1_ADM_s104
                                 + 1.17154 w_k1_AXP_s104
                                 + 0.986885 w_k1_DIS_s104
                                 + 1.01144 w_k1_FDX_s104
                                 + 1.11981 w_k1_IBM_s104
                                 + 1.3587 w_k1_INTC_s104 + 1.12272 w_k1_KO_s104
                                 + 1.22205 w_k1_MAT_s104
                                 + 1.16372 w_k1_MSFT_s104
                                 + 1.01097 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.27579 w_k1_ADM_s104
                                 + 1.08463 w_k1_AXP_s104
                                 + 1.20199 w_k1_DIS_s104
                                 + 1.16111 w_k1_FDX_s104
                                 + 1.29681 w_k1_IBM_s104
                                 + 1.24367 w_k1_INTC_s104 + 1.1784 w_k1_KO_s104
                                 + 1.13319 w_k1_MAT_s104
                                 + 0.99995 w_k1_MSFT_s104
                                 + 1.02111 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.49436 w_k1_ADM_s104
                                 + 1.60776 w_k1_AXP_s104
                                 + 1.64806 w_k1_DIS_s104
                                 + 1.57329 w_k1_FDX_s104
                                 + 1.08322 w_k1_IBM_s104
                                 + 1.46442 w_k1_INTC_s104 + 1.5691 w_k1_KO_s104
                                 + 1.37503 w_k1_MAT_s104
                                 + 1.39223 w_k1_MSFT_s104 + 1.3836 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.16798 w_k1_ADM_s105
                                 + 1.16723 w_k1_AXP_s105
                                 + 1.14459 w_k1_DIS_s105
                                 + 1.06385 w_k1_FDX_s105
                                 + 1.64131 w_k1_IBM_s105
                                 + 1.1835 w_k1_INTC_s105 + 1.18768 w_k1_KO_s105
                                 + 1.55637 w_k1_MAT_s105
                                 + 1.06249 w_k1_MSFT_s105 + 1.067 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11308 w_k1_ADM_s105
                                 + 0.968321 w_k1_AXP_s105
                                 + 0.854611 w_k1_DIS_s105
                                 + 0.94636 w_k1_FDX_s105
                                 + 0.952693 w_k1_IBM_s105
                                 + 0.881991 w_k1_INTC_s105
                                 + 0.714044 w_k1_KO_s105
                                 + 1.00239 w_k1_MAT_s105
                                 + 0.854995 w_k1_MSFT_s105
                                 + 0.93326 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.45351 w_k1_ADM_s105
                                 + 1.42177 w_k1_AXP_s105
                                 + 1.33323 w_k1_DIS_s105
                                 + 1.33346 w_k1_FDX_s105
                                 + 0.781798 w_k1_IBM_s105
                                 + 0.983112 w_k1_INTC_s105
                                 + 0.860777 w_k1_KO_s105
                                 + 1.10839 w_k1_MAT_s105
                                 + 1.09536 w_k1_MSFT_s105
                                 + 1.18718 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.850299 w_k1_ADM_s105
                                 + 0.798631 w_k1_AXP_s105
                                 + 0.551363 w_k1_DIS_s105
                                 + 0.802778 w_k1_FDX_s105
                                 + 0.995513 w_k1_IBM_s105
                                 + 0.782433 w_k1_INTC_s105
                                 + 0.61915 w_k1_KO_s105 + 1.04742 w_k1_MAT_s105
                                 + 0.756367 w_k1_MSFT_s105
                                 + 0.819918 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.76496 w_k1_ADM_s105
                                 + 2.37872 w_k1_AXP_s105
                                 + 1.97149 w_k1_DIS_s105 + 1.6222 w_k1_FDX_s105
                                 + 1.38943 w_k1_IBM_s105
                                 + 1.19521 w_k1_INTC_s105
                                 + 1.17552 w_k1_KO_s105 + 1.73326 w_k1_MAT_s105
                                 + 1.27291 w_k1_MSFT_s105
                                 + 1.26395 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.62895 w_k1_ADM_s105
                                 + 1.67566 w_k1_AXP_s105
                                 + 1.68226 w_k1_DIS_s105
                                 + 1.43626 w_k1_FDX_s105
                                 + 1.04255 w_k1_IBM_s105
                                 + 1.08872 w_k1_INTC_s105
                                 + 1.01666 w_k1_KO_s105 + 1.20398 w_k1_MAT_s105
                                 + 1.30705 w_k1_MSFT_s105
                                 + 1.23534 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.31517 w_k1_ADM_s105
                                 + 1.03045 w_k1_AXP_s105
                                 + 1.05611 w_k1_DIS_s105
                                 + 1.09856 w_k1_FDX_s105
                                 + 0.939149 w_k1_IBM_s105
                                 + 0.7799 w_k1_INTC_s105
                                 + 0.809435 w_k1_KO_s105
                                 + 0.878792 w_k1_MAT_s105
                                 + 1.03625 w_k1_MSFT_s105
                                 + 1.17807 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.00328 w_k1_ADM_s105
                                 + 0.843186 w_k1_AXP_s105
                                 + 0.740302 w_k1_DIS_s105
                                 + 1.0059 w_k1_FDX_s105
                                 + 0.575032 w_k1_IBM_s105
                                 + 0.83545 w_k1_INTC_s105
                                 + 0.571891 w_k1_KO_s105
                                 + 0.475983 w_k1_MAT_s105
                                 + 0.881838 w_k1_MSFT_s105
                                 + 0.93605 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.807785 w_k1_ADM_s106
                                 + 1.00247 w_k1_AXP_s106
                                 + 1.09855 w_k1_DIS_s106
                                 + 0.985024 w_k1_FDX_s106
                                 + 1.0989 w_k1_IBM_s106
                                 + 0.933514 w_k1_INTC_s106
                                 + 1.05199 w_k1_KO_s106 + 1.11819 w_k1_MAT_s106
                                 + 1.17571 w_k1_MSFT_s106
                                 + 1.07035 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.11987 w_k1_ADM_s106
                                 + 1.37724 w_k1_AXP_s106
                                 + 0.92185 w_k1_DIS_s106
                                 + 1.06097 w_k1_FDX_s106
                                 + 0.876876 w_k1_IBM_s106
                                 + 0.998213 w_k1_INTC_s106
                                 + 0.831948 w_k1_KO_s106
                                 + 1.00819 w_k1_MAT_s106
                                 + 1.13007 w_k1_MSFT_s106
                                 + 0.879719 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.78238 w_k1_ADM_s106
                                 + 0.917367 w_k1_AXP_s106
                                 + 1.06978 w_k1_DIS_s106
                                 + 1.02671 w_k1_FDX_s106
                                 + 0.886959 w_k1_IBM_s106
                                 + 1.03682 w_k1_INTC_s106
                                 + 0.979512 w_k1_KO_s106
                                 + 1.02245 w_k1_MAT_s106
                                 + 1.21957 w_k1_MSFT_s106 + 1.0192 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.22827 w_k1_ADM_s106
                                 + 1.24727 w_k1_AXP_s106
                                 + 1.29131 w_k1_DIS_s106
                                 + 1.08287 w_k1_FDX_s106
                                 + 1.38399 w_k1_IBM_s106
                                 + 1.0507 w_k1_INTC_s106 + 1.14567 w_k1_KO_s106
                                 + 1.67333 w_k1_MAT_s106
                                 + 1.18228 w_k1_MSFT_s106
                                 + 1.17567 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.959651 w_k1_ADM_s106
                                 + 0.774218 w_k1_AXP_s106
                                 + 0.863455 w_k1_DIS_s106
                                 + 0.882786 w_k1_FDX_s106
                                 + 1.2503 w_k1_IBM_s106
                                 + 0.907007 w_k1_INTC_s106
                                 + 1.14341 w_k1_KO_s106 + 1.39094 w_k1_MAT_s106
                                 + 0.967713 w_k1_MSFT_s106
                                 + 0.95599 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.845598 w_k1_ADM_s106
                                 + 0.877363 w_k1_AXP_s106
                                 + 0.857465 w_k1_DIS_s106
                                 + 0.95583 w_k1_FDX_s106
                                 + 0.611514 w_k1_IBM_s106
                                 + 0.911431 w_k1_INTC_s106
                                 + 0.783617 w_k1_KO_s106
                                 + 0.648327 w_k1_MAT_s106
                                 + 0.98755 w_k1_MSFT_s106
                                 + 0.746887 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.626544 w_k1_ADM_s106
                                 + 0.612287 w_k1_AXP_s106
                                 + 0.704947 w_k1_DIS_s106
                                 + 0.780271 w_k1_FDX_s106
                                 + 0.9355 w_k1_IBM_s106
                                 + 0.773775 w_k1_INTC_s106
                                 + 0.872408 w_k1_KO_s106
                                 + 0.832043 w_k1_MAT_s106
                                 + 0.804523 w_k1_MSFT_s106
                                 + 0.815398 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.11658 w_k1_ADM_s106
                                 + 1.25981 w_k1_AXP_s106
                                 + 1.48093 w_k1_DIS_s106
                                 + 1.40136 w_k1_FDX_s106
                                 + 0.875923 w_k1_IBM_s106
                                 + 1.14869 w_k1_INTC_s106
                                 + 1.16926 w_k1_KO_s106 + 1.138 w_k1_MAT_s106
                                 + 1.34605 w_k1_MSFT_s106
                                 + 1.06491 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.924009 w_k1_ADM_s107
                                 + 0.675017 w_k1_AXP_s107
                                 + 0.675475 w_k1_DIS_s107
                                 + 0.766445 w_k1_FDX_s107
                                 + 0.678819 w_k1_IBM_s107
                                 + 0.688518 w_k1_INTC_s107
                                 + 0.784322 w_k1_KO_s107
                                 + 0.815673 w_k1_MAT_s107
                                 + 0.694501 w_k1_MSFT_s107
                                 + 0.859034 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.29844 w_k1_ADM_s107
                                 + 1.26303 w_k1_AXP_s107
                                 + 1.44942 w_k1_DIS_s107
                                 + 1.13989 w_k1_FDX_s107
                                 + 0.970596 w_k1_IBM_s107
                                 + 1.02305 w_k1_INTC_s107
                                 + 1.20458 w_k1_KO_s107 + 1.10968 w_k1_MAT_s107
                                 + 1.10626 w_k1_MSFT_s107 + 0.9869 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.13614 w_k1_ADM_s107
                                 + 1.2051 w_k1_AXP_s107
                                 + 0.916129 w_k1_DIS_s107
                                 + 1.03255 w_k1_FDX_s107
                                 + 0.859825 w_k1_IBM_s107
                                 + 0.877282 w_k1_INTC_s107
                                 + 0.898948 w_k1_KO_s107
                                 + 0.852648 w_k1_MAT_s107
                                 + 0.925779 w_k1_MSFT_s107
                                 + 0.966624 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.878657 w_k1_ADM_s107
                                 + 1.31943 w_k1_AXP_s107
                                 + 0.743733 w_k1_DIS_s107
                                 + 0.869024 w_k1_FDX_s107
                                 + 0.946326 w_k1_IBM_s107
                                 + 0.89863 w_k1_INTC_s107
                                 + 0.941893 w_k1_KO_s107
                                 + 1.27043 w_k1_MAT_s107
                                 + 0.706618 w_k1_MSFT_s107
                                 + 0.939304 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.34664 w_k1_ADM_s107
                                 + 1.64747 w_k1_AXP_s107
                                 + 1.39557 w_k1_DIS_s107
                                 + 1.42385 w_k1_FDX_s107
                                 + 1.24186 w_k1_IBM_s107
                                 + 1.2689 w_k1_INTC_s107 + 1.29911 w_k1_KO_s107
                                 + 1.46371 w_k1_MAT_s107
                                 + 1.1003 w_k1_MSFT_s107 + 1.02854 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.54598 w_k1_ADM_s107
                                 + 1.67934 w_k1_AXP_s107
                                 + 1.49704 w_k1_DIS_s107
                                 + 1.16234 w_k1_FDX_s107
                                 + 1.66793 w_k1_IBM_s107
                                 + 0.977372 w_k1_INTC_s107
                                 + 1.4254 w_k1_KO_s107 + 1.82922 w_k1_MAT_s107
                                 + 1.06141 w_k1_MSFT_s107
                                 + 1.21335 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.00132 w_k1_ADM_s107
                                 + 0.906908 w_k1_AXP_s107
                                 + 0.953846 w_k1_DIS_s107
                                 + 0.878773 w_k1_FDX_s107
                                 + 1.08483 w_k1_IBM_s107
                                 + 0.868158 w_k1_INTC_s107
                                 + 0.900378 w_k1_KO_s107
                                 + 0.959524 w_k1_MAT_s107
                                 + 0.856188 w_k1_MSFT_s107
                                 + 0.922698 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.09926 w_k1_ADM_s107
                                 + 1.13836 w_k1_AXP_s107
                                 + 1.00165 w_k1_DIS_s107
                                 + 0.958127 w_k1_FDX_s107
                                 + 1.30633 w_k1_IBM_s107 + 1.069 w_k1_INTC_s107
                                 + 1.1564 w_k1_KO_s107 + 1.2024 w_k1_MAT_s107
                                 + 0.875081 w_k1_MSFT_s107
                                 + 1.01464 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.26864 w_k1_ADM_s110
                                 + 1.28515 w_k1_AXP_s110
                                 + 1.02898 w_k1_DIS_s110
                                 + 1.17008 w_k1_FDX_s110
                                 + 0.956213 w_k1_IBM_s110
                                 + 1.28269 w_k1_INTC_s110
                                 + 0.848912 w_k1_KO_s110
                                 + 1.06193 w_k1_MAT_s110
                                 + 1.04817 w_k1_MSFT_s110
                                 + 1.03797 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.32626 w_k1_ADM_s110
                                 + 1.39147 w_k1_AXP_s110
                                 + 1.51962 w_k1_DIS_s110
                                 + 1.40759 w_k1_FDX_s110
                                 + 1.32625 w_k1_IBM_s110
                                 + 1.24639 w_k1_INTC_s110
                                 + 1.23375 w_k1_KO_s110 + 1.07558 w_k1_MAT_s110
                                 + 1.22844 w_k1_MSFT_s110
                                 + 1.33694 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.888425 w_k1_ADM_s110
                                 + 1.09854 w_k1_AXP_s110
                                 + 1.18725 w_k1_DIS_s110
                                 + 1.01534 w_k1_FDX_s110
                                 + 1.62842 w_k1_IBM_s110
                                 + 1.3118 w_k1_INTC_s110 + 1.17277 w_k1_KO_s110
                                 + 1.09295 w_k1_MAT_s110
                                 + 1.1647 w_k1_MSFT_s110 + 1.09463 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.929423 w_k1_ADM_s110
                                 + 0.823227 w_k1_AXP_s110
                                 + 1.19945 w_k1_DIS_s110 + 0.9937 w_k1_FDX_s110
                                 + 1.01273 w_k1_IBM_s110
                                 + 1.00912 w_k1_INTC_s110
                                 + 1.00619 w_k1_KO_s110
                                 + 0.922562 w_k1_MAT_s110
                                 + 1.08739 w_k1_MSFT_s110
                                 + 1.03709 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.869358 w_k1_ADM_s110
                                 + 0.946857 w_k1_AXP_s110
                                 + 0.835076 w_k1_DIS_s110
                                 + 0.83628 w_k1_FDX_s110
                                 + 1.04282 w_k1_IBM_s110
                                 + 1.19111 w_k1_INTC_s110
                                 + 0.977635 w_k1_KO_s110
                                 + 0.900605 w_k1_MAT_s110
                                 + 0.948154 w_k1_MSFT_s110
                                 + 0.956929 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.963651 w_k1_ADM_s110
                                 + 0.730377 w_k1_AXP_s110
                                 + 0.76003 w_k1_DIS_s110
                                 + 0.927271 w_k1_FDX_s110
                                 + 0.697344 w_k1_IBM_s110
                                 + 1.12732 w_k1_INTC_s110
                                 + 0.768528 w_k1_KO_s110
                                 + 0.510378 w_k1_MAT_s110
                                 + 0.846445 w_k1_MSFT_s110
                                 + 0.890694 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.96372 w_k1_ADM_s110
                                 + 0.95424 w_k1_AXP_s110
                                 + 1.52832 w_k1_DIS_s110
                                 + 1.12303 w_k1_FDX_s110
                                 + 1.13043 w_k1_IBM_s110
                                 + 1.23081 w_k1_INTC_s110
                                 + 0.98584 w_k1_KO_s110
                                 + 0.776689 w_k1_MAT_s110
                                 + 1.27659 w_k1_MSFT_s110 + 1.0192 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.00665 w_k1_ADM_s110
                                 + 0.976585 w_k1_AXP_s110
                                 + 1.25397 w_k1_DIS_s110
                                 + 1.22779 w_k1_FDX_s110
                                 + 0.762988 w_k1_IBM_s110
                                 + 1.43121 w_k1_INTC_s110
                                 + 1.01146 w_k1_KO_s110 + 0.74209 w_k1_MAT_s110
                                 + 1.16159 w_k1_MSFT_s110
                                 + 1.16721 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.799704 w_k1_ADM_s111
                                 + 0.756936 w_k1_AXP_s111
                                 + 0.697222 w_k1_DIS_s111
                                 + 0.848326 w_k1_FDX_s111
                                 + 0.8854 w_k1_IBM_s111
                                 + 0.673341 w_k1_INTC_s111
                                 + 0.798488 w_k1_KO_s111
                                 + 0.852429 w_k1_MAT_s111
                                 + 0.741085 w_k1_MSFT_s111
                                 + 0.886022 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.892112 w_k1_ADM_s111
                                 + 0.989927 w_k1_AXP_s111
                                 + 0.99398 w_k1_DIS_s111
                                 + 1.00562 w_k1_FDX_s111
                                 + 1.06583 w_k1_IBM_s111
                                 + 1.00971 w_k1_INTC_s111
                                 + 0.981111 w_k1_KO_s111
                                 + 0.958141 w_k1_MAT_s111
                                 + 1.03995 w_k1_MSFT_s111
                                 + 0.971419 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.11393 w_k1_ADM_s111
                                 + 1.3961 w_k1_AXP_s111 + 1.36043 w_k1_DIS_s111
                                 + 1.30049 w_k1_FDX_s111
                                 + 1.25006 w_k1_IBM_s111
                                 + 1.17434 w_k1_INTC_s111
                                 + 1.27001 w_k1_KO_s111 + 1.46531 w_k1_MAT_s111
                                 + 1.15526 w_k1_MSFT_s111
                                 + 1.18675 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.767261 w_k1_ADM_s111
                                 + 1.00572 w_k1_AXP_s111
                                 + 0.969698 w_k1_DIS_s111
                                 + 0.914426 w_k1_FDX_s111
                                 + 1.37188 w_k1_IBM_s111
                                 + 0.80257 w_k1_INTC_s111
                                 + 1.03785 w_k1_KO_s111 + 1.18999 w_k1_MAT_s111
                                 + 0.838219 w_k1_MSFT_s111
                                 + 1.03483 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04099 w_k1_ADM_s111
                                 + 0.798363 w_k1_AXP_s111
                                 + 1.20489 w_k1_DIS_s111
                                 + 1.17688 w_k1_FDX_s111
                                 + 0.710172 w_k1_IBM_s111
                                 + 1.02735 w_k1_INTC_s111
                                 + 0.888457 w_k1_KO_s111
                                 + 0.638447 w_k1_MAT_s111
                                 + 1.16363 w_k1_MSFT_s111
                                 + 0.963162 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.03423 w_k1_ADM_s111
                                 + 1.02526 w_k1_AXP_s111
                                 + 0.820246 w_k1_DIS_s111
                                 + 1.09252 w_k1_FDX_s111
                                 + 0.850185 w_k1_IBM_s111
                                 + 0.890209 w_k1_INTC_s111
                                 + 0.798141 w_k1_KO_s111
                                 + 1.00907 w_k1_MAT_s111
                                 + 1.04091 w_k1_MSFT_s111
                                 + 0.995861 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.01058 w_k1_ADM_s111
                                 + 0.952658 w_k1_AXP_s111
                                 + 1.46047 w_k1_DIS_s111 + 1.1297 w_k1_FDX_s111
                                 + 1.24448 w_k1_IBM_s111
                                 + 0.972007 w_k1_INTC_s111
                                 + 1.15952 w_k1_KO_s111
                                 + 0.813271 w_k1_MAT_s111
                                 + 1.14861 w_k1_MSFT_s111
                                 + 1.10259 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.957926 w_k1_ADM_s111
                                 + 0.700644 w_k1_AXP_s111
                                 + 1.13888 w_k1_DIS_s111
                                 + 0.974407 w_k1_FDX_s111
                                 + 0.799831 w_k1_IBM_s111
                                 + 1.04845 w_k1_INTC_s111
                                 + 0.987542 w_k1_KO_s111
                                 + 0.674656 w_k1_MAT_s111
                                 + 0.95261 w_k1_MSFT_s111
                                 + 1.00186 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.08285 w_k1_ADM_s112
                                 + 1.00774 w_k1_AXP_s112
                                 + 0.609517 w_k1_DIS_s112
                                 + 0.79077 w_k1_FDX_s112
                                 + 0.98661 w_k1_IBM_s112
                                 + 0.759378 w_k1_INTC_s112
                                 + 0.785577 w_k1_KO_s112
                                 + 1.05655 w_k1_MAT_s112
                                 + 0.750773 w_k1_MSFT_s112
                                 + 0.818852 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.57758 w_k1_ADM_s112
                                 + 1.347 w_k1_AXP_s112 + 1.76134 w_k1_DIS_s112
                                 + 1.43982 w_k1_FDX_s112
                                 + 1.54567 w_k1_IBM_s112
                                 + 1.35132 w_k1_INTC_s112
                                 + 1.43497 w_k1_KO_s112 + 1.35613 w_k1_MAT_s112
                                 + 1.47284 w_k1_MSFT_s112
                                 + 1.16284 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.25136 w_k1_ADM_s112
                                 + 1.33516 w_k1_AXP_s112
                                 + 1.11574 w_k1_DIS_s112
                                 + 1.24512 w_k1_FDX_s112 + 1.0778 w_k1_IBM_s112
                                 + 0.969984 w_k1_INTC_s112
                                 + 0.96498 w_k1_KO_s112 + 1.1404 w_k1_MAT_s112
                                 + 1.14793 w_k1_MSFT_s112
                                 + 1.04534 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.35045 w_k1_ADM_s112
                                 + 1.17368 w_k1_AXP_s112
                                 + 1.29875 w_k1_DIS_s112
                                 + 1.04371 w_k1_FDX_s112
                                 + 1.48198 w_k1_IBM_s112
                                 + 0.802565 w_k1_INTC_s112
                                 + 1.17716 w_k1_KO_s112 + 2.06426 w_k1_MAT_s112
                                 + 1.05507 w_k1_MSFT_s112
                                 + 1.13123 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.34127 w_k1_ADM_s112
                                 + 1.01654 w_k1_AXP_s112
                                 + 0.927843 w_k1_DIS_s112
                                 + 1.01894 w_k1_FDX_s112
                                 + 1.02773 w_k1_IBM_s112
                                 + 0.844794 w_k1_INTC_s112
                                 + 0.974964 w_k1_KO_s112
                                 + 1.10024 w_k1_MAT_s112
                                 + 0.996204 w_k1_MSFT_s112
                                 + 0.993461 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.904113 w_k1_ADM_s112
                                 + 0.706939 w_k1_AXP_s112
                                 + 0.949398 w_k1_DIS_s112
                                 + 0.818731 w_k1_FDX_s112
                                 + 1.25379 w_k1_IBM_s112
                                 + 0.735268 w_k1_INTC_s112
                                 + 0.981837 w_k1_KO_s112
                                 + 1.00187 w_k1_MAT_s112
                                 + 0.797642 w_k1_MSFT_s112
                                 + 0.914533 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.06435 w_k1_ADM_s112
                                 + 1.11724 w_k1_AXP_s112
                                 + 1.38643 w_k1_DIS_s112 + 1.1132 w_k1_FDX_s112
                                 + 0.965865 w_k1_IBM_s112
                                 + 0.895133 w_k1_INTC_s112
                                 + 1.05399 w_k1_KO_s112 + 1.15037 w_k1_MAT_s112
                                 + 1.04099 w_k1_MSFT_s112
                                 + 1.01416 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.29151 w_k1_ADM_s112
                                 + 0.770751 w_k1_AXP_s112
                                 + 0.737451 w_k1_DIS_s112
                                 + 0.878237 w_k1_FDX_s112
                                 + 0.664994 w_k1_IBM_s112
                                 + 0.819799 w_k1_INTC_s112
                                 + 0.733137 w_k1_KO_s112
                                 + 0.55133 w_k1_MAT_s112
                                 + 0.826853 w_k1_MSFT_s112
                                 + 0.789334 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.09698 w_k1_ADM_s113
                                 + 1.13738 w_k1_AXP_s113
                                 + 0.974999 w_k1_DIS_s113
                                 + 1.11172 w_k1_FDX_s113
                                 + 0.866177 w_k1_IBM_s113
                                 + 0.879565 w_k1_INTC_s113
                                 + 1.0166 w_k1_KO_s113 + 1.32957 w_k1_MAT_s113
                                 + 0.996654 w_k1_MSFT_s113
                                 + 0.983161 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.23276 w_k1_ADM_s113
                                 + 0.943747 w_k1_AXP_s113
                                 + 1.25066 w_k1_DIS_s113
                                 + 1.17326 w_k1_FDX_s113
                                 + 0.79544 w_k1_IBM_s113
                                 + 0.919355 w_k1_INTC_s113
                                 + 1.10451 w_k1_KO_s113
                                 + 0.734448 w_k1_MAT_s113
                                 + 1.16714 w_k1_MSFT_s113
                                 + 1.04156 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.893392 w_k1_ADM_s113
                                 + 1.00649 w_k1_AXP_s113
                                 + 0.842474 w_k1_DIS_s113
                                 + 0.891119 w_k1_FDX_s113
                                 + 1.16364 w_k1_IBM_s113
                                 + 0.954612 w_k1_INTC_s113
                                 + 1.00856 w_k1_KO_s113 + 1.08234 w_k1_MAT_s113
                                 + 0.900421 w_k1_MSFT_s113
                                 + 0.992555 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.823838 w_k1_ADM_s113
                                 + 0.606492 w_k1_AXP_s113
                                 + 0.621698 w_k1_DIS_s113
                                 + 0.731597 w_k1_FDX_s113
                                 + 0.817563 w_k1_IBM_s113
                                 + 0.644383 w_k1_INTC_s113
                                 + 0.738708 w_k1_KO_s113
                                 + 0.773749 w_k1_MAT_s113
                                 + 0.634005 w_k1_MSFT_s113
                                 + 0.782549 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.966104 w_k1_ADM_s113
                                 + 0.814822 w_k1_AXP_s113
                                 + 1.01154 w_k1_DIS_s113
                                 + 0.930549 w_k1_FDX_s113
                                 + 0.797228 w_k1_IBM_s113
                                 + 0.921714 w_k1_INTC_s113
                                 + 1.01991 w_k1_KO_s113
                                 + 0.511443 w_k1_MAT_s113
                                 + 0.964969 w_k1_MSFT_s113
                                 + 0.950279 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.944888 w_k1_ADM_s113
                                 + 0.942163 w_k1_AXP_s113
                                 + 0.648934 w_k1_DIS_s113
                                 + 0.855623 w_k1_FDX_s113
                                 + 0.722003 w_k1_IBM_s113
                                 + 0.720651 w_k1_INTC_s113
                                 + 0.818954 w_k1_KO_s113
                                 + 0.787628 w_k1_MAT_s113
                                 + 0.792231 w_k1_MSFT_s113
                                 + 0.826409 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.915683 w_k1_ADM_s113
                                 + 1.12259 w_k1_AXP_s113
                                 + 1.54922 w_k1_DIS_s113
                                 + 1.21326 w_k1_FDX_s113
                                 + 1.07378 w_k1_IBM_s113
                                 + 1.19776 w_k1_INTC_s113
                                 + 1.28884 w_k1_KO_s113 + 1.06588 w_k1_MAT_s113
                                 + 1.23911 w_k1_MSFT_s113 + 1.1653 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.0859 w_k1_ADM_s113
                                 + 1.4083 w_k1_AXP_s113 + 1.64547 w_k1_DIS_s113
                                 + 1.21202 w_k1_FDX_s113
                                 + 1.83385 w_k1_IBM_s113
                                 + 1.14287 w_k1_INTC_s113 + 1.6435 w_k1_KO_s113
                                 + 1.76563 w_k1_MAT_s113
                                 + 1.27483 w_k1_MSFT_s113
                                 + 1.20969 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.32703 w_k1_ADM_s114
                                 + 1.25789 w_k1_AXP_s114
                                 + 1.82709 w_k1_DIS_s114
                                 + 1.20693 w_k1_FDX_s114
                                 + 1.61077 w_k1_IBM_s114
                                 + 1.19867 w_k1_INTC_s114
                                 + 1.29556 w_k1_KO_s114 + 1.40037 w_k1_MAT_s114
                                 + 1.21325 w_k1_MSFT_s114
                                 + 1.29887 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.852315 w_k1_ADM_s114
                                 + 0.735454 w_k1_AXP_s114
                                 + 0.727626 w_k1_DIS_s114
                                 + 0.802641 w_k1_FDX_s114
                                 + 0.705364 w_k1_IBM_s114
                                 + 1.14158 w_k1_INTC_s114
                                 + 0.769549 w_k1_KO_s114
                                 + 0.681861 w_k1_MAT_s114
                                 + 0.867422 w_k1_MSFT_s114
                                 + 0.813958 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.01448 w_k1_ADM_s114
                                 + 1.10713 w_k1_AXP_s114
                                 + 0.899692 w_k1_DIS_s114
                                 + 0.920469 w_k1_FDX_s114
                                 + 1.22253 w_k1_IBM_s114
                                 + 0.968011 w_k1_INTC_s114
                                 + 0.972678 w_k1_KO_s114
                                 + 1.54047 w_k1_MAT_s114
                                 + 0.855406 w_k1_MSFT_s114
                                 + 1.1505 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.05929 w_k1_ADM_s114
                                 + 0.96646 w_k1_AXP_s114
                                 + 1.12183 w_k1_DIS_s114 + 1.1295 w_k1_FDX_s114
                                 + 0.888614 w_k1_IBM_s114
                                 + 1.20536 w_k1_INTC_s114
                                 + 1.11177 w_k1_KO_s114 + 1.04657 w_k1_MAT_s114
                                 + 1.09234 w_k1_MSFT_s114
                                 + 1.08177 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14491 w_k1_ADM_s114
                                 + 1.14978 w_k1_AXP_s114
                                 + 1.19276 w_k1_DIS_s114
                                 + 1.02028 w_k1_FDX_s114
                                 + 1.43542 w_k1_IBM_s114
                                 + 1.25825 w_k1_INTC_s114
                                 + 1.27648 w_k1_KO_s114 + 1.0501 w_k1_MAT_s114
                                 + 1.01584 w_k1_MSFT_s114
                                 + 1.07749 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.43859 w_k1_ADM_s114
                                 + 1.51881 w_k1_AXP_s114
                                 + 1.48368 w_k1_DIS_s114
                                 + 1.39595 w_k1_FDX_s114
                                 + 1.23557 w_k1_IBM_s114
                                 + 1.51245 w_k1_INTC_s114
                                 + 1.23139 w_k1_KO_s114 + 1.16177 w_k1_MAT_s114
                                 + 1.24406 w_k1_MSFT_s114
                                 + 1.19188 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.13805 w_k1_ADM_s114
                                 + 0.938958 w_k1_AXP_s114
                                 + 1.07883 w_k1_DIS_s114
                                 + 1.07224 w_k1_FDX_s114
                                 + 0.911662 w_k1_IBM_s114
                                 + 1.15696 w_k1_INTC_s114
                                 + 0.959615 w_k1_KO_s114
                                 + 0.960243 w_k1_MAT_s114
                                 + 0.991443 w_k1_MSFT_s114
                                 + 0.986868 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.985247 w_k1_ADM_s114
                                 + 0.876417 w_k1_AXP_s114
                                 + 0.754429 w_k1_DIS_s114
                                 + 0.84873 w_k1_FDX_s114
                                 + 0.968403 w_k1_IBM_s114
                                 + 0.890943 w_k1_INTC_s114
                                 + 0.896932 w_k1_KO_s114
                                 + 0.921892 w_k1_MAT_s114
                                 + 0.824259 w_k1_MSFT_s114
                                 + 1.00745 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.980093 w_k1_ADM_s115
                                 + 0.962807 w_k1_AXP_s115
                                 + 0.78701 w_k1_DIS_s115
                                 + 0.928672 w_k1_FDX_s115
                                 + 0.845961 w_k1_IBM_s115
                                 + 1.20265 w_k1_INTC_s115
                                 + 0.873528 w_k1_KO_s115
                                 + 1.12959 w_k1_MAT_s115
                                 + 0.833602 w_k1_MSFT_s115
                                 + 0.885609 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.00729 w_k1_ADM_s115
                                 + 0.909034 w_k1_AXP_s115
                                 + 0.688705 w_k1_DIS_s115
                                 + 0.806857 w_k1_FDX_s115
                                 + 1.24446 w_k1_IBM_s115
                                 + 0.866436 w_k1_INTC_s115
                                 + 0.940959 w_k1_KO_s115
                                 + 1.22105 w_k1_MAT_s115
                                 + 0.763812 w_k1_MSFT_s115
                                 + 0.986755 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.895396 w_k1_ADM_s115
                                 + 1.02211 w_k1_AXP_s115
                                 + 0.813072 w_k1_DIS_s115
                                 + 0.96887 w_k1_FDX_s115
                                 + 0.676398 w_k1_IBM_s115
                                 + 0.953019 w_k1_INTC_s115
                                 + 0.745804 w_k1_KO_s115
                                 + 0.784065 w_k1_MAT_s115
                                 + 0.85311 w_k1_MSFT_s115
                                 + 0.870645 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.03664 w_k1_ADM_s115
                                 + 1.10157 w_k1_AXP_s115
                                 + 1.16077 w_k1_DIS_s115
                                 + 1.11402 w_k1_FDX_s115
                                 + 1.23872 w_k1_IBM_s115
                                 + 1.03152 w_k1_INTC_s115
                                 + 1.21749 w_k1_KO_s115 + 1.46911 w_k1_MAT_s115
                                 + 1.09618 w_k1_MSFT_s115
                                 + 1.11638 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.14999 w_k1_ADM_s115
                                 + 1.02437 w_k1_AXP_s115
                                 + 1.10096 w_k1_DIS_s115 + 1.2032 w_k1_FDX_s115
                                 + 0.785032 w_k1_IBM_s115
                                 + 1.17719 w_k1_INTC_s115
                                 + 1.01518 w_k1_KO_s115 + 1.08397 w_k1_MAT_s115
                                 + 1.16611 w_k1_MSFT_s115
                                 + 1.11939 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.85979 w_k1_ADM_s115
                                 + 0.689431 w_k1_AXP_s115
                                 + 0.894212 w_k1_DIS_s115
                                 + 0.917679 w_k1_FDX_s115
                                 + 0.787227 w_k1_IBM_s115
                                 + 0.998791 w_k1_INTC_s115
                                 + 0.859072 w_k1_KO_s115
                                 + 0.807353 w_k1_MAT_s115
                                 + 0.853702 w_k1_MSFT_s115
                                 + 0.868151 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.43384 w_k1_ADM_s115
                                 + 1.92007 w_k1_AXP_s115
                                 + 1.72432 w_k1_DIS_s115
                                 + 1.44803 w_k1_FDX_s115
                                 + 1.55537 w_k1_IBM_s115
                                 + 1.39692 w_k1_INTC_s115
                                 + 1.35252 w_k1_KO_s115 + 2.05488 w_k1_MAT_s115
                                 + 1.1834 w_k1_MSFT_s115 + 1.29874 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.16806 w_k1_ADM_s115
                                 + 1.32329 w_k1_AXP_s115
                                 + 1.51157 w_k1_DIS_s115
                                 + 1.30337 w_k1_FDX_s115
                                 + 1.10879 w_k1_IBM_s115
                                 + 1.15744 w_k1_INTC_s115
                                 + 1.36037 w_k1_KO_s115 + 1.43272 w_k1_MAT_s115
                                 + 1.20901 w_k1_MSFT_s115
                                 + 1.14081 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.972075 w_k1_ADM_s116
                                 + 0.890806 w_k1_AXP_s116
                                 + 0.842356 w_k1_DIS_s116
                                 + 1.01228 w_k1_FDX_s116
                                 + 0.566885 w_k1_IBM_s116
                                 + 0.921854 w_k1_INTC_s116
                                 + 0.684941 w_k1_KO_s116
                                 + 0.645774 w_k1_MAT_s116
                                 + 0.980913 w_k1_MSFT_s116
                                 + 0.848976 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.23424 w_k1_ADM_s116
                                 + 1.43494 w_k1_AXP_s116 + 1.1979 w_k1_DIS_s116
                                 + 1.07797 w_k1_FDX_s116
                                 + 1.12104 w_k1_IBM_s116
                                 + 1.1844 w_k1_INTC_s116 + 1.19108 w_k1_KO_s116
                                 + 1.43675 w_k1_MAT_s116
                                 + 0.965389 w_k1_MSFT_s116
                                 + 1.11991 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.657099 w_k1_ADM_s116
                                 + 0.452914 w_k1_AXP_s116
                                 + 0.508387 w_k1_DIS_s116
                                 + 0.594045 w_k1_FDX_s116
                                 + 0.678769 w_k1_IBM_s116
                                 + 0.62891 w_k1_INTC_s116
                                 + 0.704247 w_k1_KO_s116
                                 + 0.453917 w_k1_MAT_s116
                                 + 0.614915 w_k1_MSFT_s116
                                 + 0.722548 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.897009 w_k1_ADM_s116
                                 + 0.990791 w_k1_AXP_s116
                                 + 1.23356 w_k1_DIS_s116
                                 + 1.00374 w_k1_FDX_s116
                                 + 1.02921 w_k1_IBM_s116
                                 + 0.939735 w_k1_INTC_s116
                                 + 1.14371 w_k1_KO_s116 + 1.07406 w_k1_MAT_s116
                                 + 1.01064 w_k1_MSFT_s116
                                 + 0.942729 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.752883 w_k1_ADM_s116
                                 + 0.891241 w_k1_AXP_s116
                                 + 0.627379 w_k1_DIS_s116
                                 + 0.792509 w_k1_FDX_s116
                                 + 0.666148 w_k1_IBM_s116
                                 + 0.884315 w_k1_INTC_s116
                                 + 0.777685 w_k1_KO_s116
                                 + 0.999628 w_k1_MAT_s116
                                 + 0.724552 w_k1_MSFT_s116
                                 + 0.856952 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.19549 w_k1_ADM_s116
                                 + 1.01493 w_k1_AXP_s116
                                 + 1.20559 w_k1_DIS_s116
                                 + 1.12027 w_k1_FDX_s116
                                 + 0.755957 w_k1_IBM_s116
                                 + 1.08169 w_k1_INTC_s116
                                 + 0.907648 w_k1_KO_s116
                                 + 0.71798 w_k1_MAT_s116
                                 + 1.06162 w_k1_MSFT_s116
                                 + 0.948554 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.8941 w_k1_ADM_s116
                                 + 0.821863 w_k1_AXP_s116
                                 + 0.683594 w_k1_DIS_s116
                                 + 0.787989 w_k1_FDX_s116
                                 + 0.909054 w_k1_IBM_s116
                                 + 0.683544 w_k1_INTC_s116
                                 + 0.860368 w_k1_KO_s116
                                 + 1.05977 w_k1_MAT_s116
                                 + 0.78446 w_k1_MSFT_s116
                                 + 0.954147 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.883666 w_k1_ADM_s116
                                 + 1.35973 w_k1_AXP_s116
                                 + 1.41564 w_k1_DIS_s116
                                 + 1.05081 w_k1_FDX_s116
                                 + 1.53765 w_k1_IBM_s116
                                 + 0.948309 w_k1_INTC_s116
                                 + 1.23118 w_k1_KO_s116 + 1.90488 w_k1_MAT_s116
                                 + 1.0504 w_k1_MSFT_s116 + 1.1315 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.40919 w_k1_ADM_s117
                                 + 1.29417 w_k1_AXP_s117
                                 + 1.46047 w_k1_DIS_s117
                                 + 1.05817 w_k1_FDX_s117 + 1.5923 w_k1_IBM_s117
                                 + 1.03484 w_k1_INTC_s117
                                 + 1.40793 w_k1_KO_s117 + 1.62876 w_k1_MAT_s117
                                 + 1.15104 w_k1_MSFT_s117
                                 + 1.24322 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.39134 w_k1_ADM_s117
                                 + 1.07148 w_k1_AXP_s117
                                 + 1.51531 w_k1_DIS_s117
                                 + 1.18687 w_k1_FDX_s117
                                 + 1.09424 w_k1_IBM_s117
                                 + 1.15422 w_k1_INTC_s117
                                 + 1.34117 w_k1_KO_s117 + 1.16617 w_k1_MAT_s117
                                 + 1.16668 w_k1_MSFT_s117
                                 + 1.22484 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.915766 w_k1_ADM_s117
                                 + 0.990411 w_k1_AXP_s117
                                 + 0.933357 w_k1_DIS_s117
                                 + 0.882582 w_k1_FDX_s117
                                 + 1.23532 w_k1_IBM_s117
                                 + 0.974104 w_k1_INTC_s117
                                 + 0.991446 w_k1_KO_s117
                                 + 0.993221 w_k1_MAT_s117
                                 + 0.811896 w_k1_MSFT_s117
                                 + 0.977675 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.13385 w_k1_ADM_s117
                                 + 0.824244 w_k1_AXP_s117
                                 + 1.1728 w_k1_DIS_s117 + 0.99982 w_k1_FDX_s117
                                 + 1.16615 w_k1_IBM_s117
                                 + 1.06501 w_k1_INTC_s117
                                 + 1.27596 w_k1_KO_s117
                                 + 0.808516 w_k1_MAT_s117
                                 + 0.995515 w_k1_MSFT_s117
                                 + 1.1561 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.31853 w_k1_ADM_s117
                                 + 1.11641 w_k1_AXP_s117
                                 + 1.05789 w_k1_DIS_s117
                                 + 1.17585 w_k1_FDX_s117
                                 + 0.904178 w_k1_IBM_s117
                                 + 1.07818 w_k1_INTC_s117
                                 + 0.947382 w_k1_KO_s117
                                 + 0.916308 w_k1_MAT_s117
                                 + 1.01077 w_k1_MSFT_s117
                                 + 1.09615 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.744228 w_k1_ADM_s117
                                 + 0.608005 w_k1_AXP_s117
                                 + 0.7104 w_k1_DIS_s117
                                 + 0.773735 w_k1_FDX_s117
                                 + 0.882345 w_k1_IBM_s117
                                 + 0.886462 w_k1_INTC_s117
                                 + 0.842883 w_k1_KO_s117
                                 + 0.743214 w_k1_MAT_s117
                                 + 0.907925 w_k1_MSFT_s117
                                 + 0.915053 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.43662 w_k1_ADM_s117
                                 + 1.58031 w_k1_AXP_s117
                                 + 1.57482 w_k1_DIS_s117 + 1.3808 w_k1_FDX_s117
                                 + 1.22067 w_k1_IBM_s117
                                 + 1.36146 w_k1_INTC_s117
                                 + 1.19891 w_k1_KO_s117 + 1.09765 w_k1_MAT_s117
                                 + 1.16036 w_k1_MSFT_s117
                                 + 1.15909 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.03435 w_k1_ADM_s117
                                 + 0.710149 w_k1_AXP_s117
                                 + 0.812444 w_k1_DIS_s117
                                 + 0.944245 w_k1_FDX_s117
                                 + 0.733966 w_k1_IBM_s117
                                 + 0.803074 w_k1_INTC_s117
                                 + 0.757234 w_k1_KO_s117
                                 + 0.540133 w_k1_MAT_s117
                                 + 0.754247 w_k1_MSFT_s117
                                 + 0.861249 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0442379453155433 w_k1_ADM_s000
                                 + 0.0155085052744067 w_k1_ADM_s001
                                 + 0.0451547744255145 w_k1_ADM_s002
                                 + 0.0531963435349628 w_k1_ADM_s003
                                 + 0.0214704775744808 w_k1_ADM_s004
                                 + 0.0222481844730081 w_k1_ADM_s005
                                 + 0.0211811722254668 w_k1_ADM_s006
                                 + 0.0272353674414579 w_k1_ADM_s007
                                 + 0.0232631949311108 w_k1_ADM_s010
                                 + 0.0165089713523098 w_k1_ADM_s011
                                 + 0.0135511594919398 w_k1_ADM_s012
                                 + 0.0243858938562004 w_k1_ADM_s013
                                 + 0.00471060733103584 w_k1_ADM_s014
                                 + 0.0206066378969728 w_k1_ADM_s015
                                 + 0.0109855431928808 w_k1_ADM_s016
                                 + 0.00713018330063013 w_k1_ADM_s017
                                 + 0.0346296658486782 w_k1_ADM_s100
                                 + 0.0304617358077587 w_k1_ADM_s101
                                 + 0.053375268280801 w_k1_ADM_s102
                                 + 0.0465464530736014 w_k1_ADM_s103
                                 + 0.0150612322188213 w_k1_ADM_s104
                                 + 0.0310808901622198 w_k1_ADM_s105
                                 + 0.017173245301002 w_k1_ADM_s106
                                 + 0.0189024212678821 w_k1_ADM_s107
                                 + 0.0792527398737574 w_k1_ADM_s110
                                 + 0.021156434787785 w_k1_ADM_s111
                                 + 0.0228927858808544 w_k1_ADM_s112
                                 + 0.0269014786015614 w_k1_ADM_s113
                                 + 0.0497803672164688 w_k1_ADM_s114
                                 + 0.070580217244999 w_k1_ADM_s115
                                 + 0.0818923562645695 w_k1_ADM_s116
                                 + 0.0498368865817692 w_k1_ADM_s117
                                 + 0.0451757738201177 w_k1_AXP_s000
                                 + 0.0153293007260756 w_k1_AXP_s001
                                 + 0.0477727750621751 w_k1_AXP_s002
                                 + 0.0477718396578418 w_k1_AXP_s003
                                 + 0.0215939075053978 w_k1_AXP_s004
                                 + 0.0215204427499861 w_k1_AXP_s005
                                 + 0.0233403575859916 w_k1_AXP_s006
                                 + 0.0307560492066138 w_k1_AXP_s007
                                 + 0.0237279269008171 w_k1_AXP_s010
                                 + 0.0151968357820056 w_k1_AXP_s011
                                 + 0.0157710493372305 w_k1_AXP_s012
                                 + 0.026458509136782 w_k1_AXP_s013
                                 + 0.00591117509459932 w_k1_AXP_s014
                                 + 0.0208917063007464 w_k1_AXP_s015
                                 + 0.0110721516179045 w_k1_AXP_s016
                                 + 0.0071088957445848 w_k1_AXP_s017
                                 + 0.0313821785824556 w_k1_AXP_s100
                                 + 0.0330505609294197 w_k1_AXP_s101
                                 + 0.0558840821357902 w_k1_AXP_s102
                                 + 0.0516453971293145 w_k1_AXP_s103
                                 + 0.0166857989860779 w_k1_AXP_s104
                                 + 0.0296398757399544 w_k1_AXP_s105
                                 + 0.0183664301079201 w_k1_AXP_s106
                                 + 0.0194832938412964 w_k1_AXP_s107
                                 + 0.0782465490935397 w_k1_AXP_s110
                                 + 0.0215284345574936 w_k1_AXP_s111
                                 + 0.0187023961652431 w_k1_AXP_s112
                                 + 0.027658969603997 w_k1_AXP_s113
                                 + 0.0470199929126329 w_k1_AXP_s114
                                 + 0.0723756432564912 w_k1_AXP_s115
                                 + 0.0915354290108825 w_k1_AXP_s116
                                 + 0.0418794995240316 w_k1_AXP_s117
                                 + 0.0441693429664511 w_k1_DIS_s000
                                 + 0.0179633212464665 w_k1_DIS_s001
                                 + 0.0467030603971112 w_k1_DIS_s002
                                 + 0.0519305103067426 w_k1_DIS_s003
                                 + 0.0210262040567506 w_k1_DIS_s004
                                 + 0.0228055253700664 w_k1_DIS_s005
                                 + 0.0233594240349864 w_k1_DIS_s006
                                 + 0.0290429427193725 w_k1_DIS_s007
                                 + 0.0243519215131536 w_k1_DIS_s010
                                 + 0.0163553658994698 w_k1_DIS_s011
                                 + 0.0160270947678662 w_k1_DIS_s012
                                 + 0.0268121504539451 w_k1_DIS_s013
                                 + 0.00590335845182071 w_k1_DIS_s014
                                 + 0.0216067991402925 w_k1_DIS_s015
                                 + 0.0113821346653804 w_k1_DIS_s016
                                 + 0.00714676572484881 w_k1_DIS_s017
                                 + 0.0341119302145248 w_k1_DIS_s100
                                 + 0.034098750935532 w_k1_DIS_s101
                                 + 0.0539552082283013 w_k1_DIS_s102
                                 + 0.0488124995357595 w_k1_DIS_s103
                                 + 0.015602570169577 w_k1_DIS_s104
                                 + 0.0266402224847022 w_k1_DIS_s105
                                 + 0.0196538982130893 w_k1_DIS_s106
                                 + 0.0174565928530091 w_k1_DIS_s107
                                 + 0.0822375504863956 w_k1_DIS_s110
                                 + 0.0235091657936035 w_k1_DIS_s111
                                 + 0.019231920136556 w_k1_DIS_s112
                                 + 0.029071474339625 w_k1_DIS_s113
                                 + 0.0495598057438864 w_k1_DIS_s114
                                 + 0.0710452612723086 w_k1_DIS_s115
                                 + 0.0909690025465135 w_k1_DIS_s116
                                 + 0.046271599245851 w_k1_DIS_s117
                                 + 0.0449535157187573 w_k1_FDX_s000
                                 + 0.0153213317246599 w_k1_FDX_s001
                                 + 0.0443852573997375 w_k1_FDX_s002
                                 + 0.0491205613222767 w_k1_FDX_s003
                                 + 0.0212809980241276 w_k1_FDX_s004
                                 + 0.0224379193535011 w_k1_FDX_s005
                                 + 0.0232841985748207 w_k1_FDX_s006
                                 + 0.0290831359071923 w_k1_FDX_s007
                                 + 0.0238282040511394 w_k1_FDX_s010
                                 + 0.0154774797009389 w_k1_FDX_s011
                                 + 0.0146674338751287 w_k1_FDX_s012
                                 + 0.0268894838592432 w_k1_FDX_s013
                                 + 0.00565126862399294 w_k1_FDX_s014
                                 + 0.0202657676337729 w_k1_FDX_s015
                                 + 0.0106274261692496 w_k1_FDX_s016
                                 + 0.00677544491556596 w_k1_FDX_s017
                                 + 0.033499784216776 w_k1_FDX_s100
                                 + 0.0326890459077943 w_k1_FDX_s101
                                 + 0.0550010796910934 w_k1_FDX_s102
                                 + 0.0477513890618173 w_k1_FDX_s103
                                 + 0.0156849720391686 w_k1_FDX_s104
                                 + 0.0279662815223459 w_k1_FDX_s105
                                 + 0.0189391813384724 w_k1_FDX_s106
                                 + 0.0170080130273411 w_k1_FDX_s107
                                 + 0.0807800447568169 w_k1_FDX_s110
                                 + 0.0233075308333168 w_k1_FDX_s111
                                 + 0.0187723128108632 w_k1_FDX_s112
                                 + 0.027672884896261 w_k1_FDX_s113
                                 + 0.046644518752253 w_k1_FDX_s114
                                 + 0.0719275359982831 w_k1_FDX_s115
                                 + 0.0823223805423402 w_k1_FDX_s116
                                 + 0.0448358665052514 w_k1_FDX_s117
                                 + 0.044613252948049 w_k1_IBM_s000
                                 + 0.015945368254936 w_k1_IBM_s001
                                 + 0.0441065666858591 w_k1_IBM_s002
                                 + 0.0491806404053085 w_k1_IBM_s003
                                 + 0.0210839672324559 w_k1_IBM_s004
                                 + 0.0241994488393233 w_k1_IBM_s005
                                 + 0.0238416295412953 w_k1_IBM_s006
                                 + 0.030338620438703 w_k1_IBM_s007
                                 + 0.0244479317053553 w_k1_IBM_s010
                                 + 0.0157104200095901 w_k1_IBM_s011
                                 + 0.0146367998855947 w_k1_IBM_s012
                                 + 0.0264727228335174 w_k1_IBM_s013
                                 + 0.00586361850629057 w_k1_IBM_s014
                                 + 0.020860375091656 w_k1_IBM_s015
                                 + 0.0110654415405632 w_k1_IBM_s016
                                 + 0.00720016083259543 w_k1_IBM_s017
                                 + 0.0336697104351948 w_k1_IBM_s100
                                 + 0.0325678705384019 w_k1_IBM_s101
                                 + 0.0554835172302827 w_k1_IBM_s102
                                 + 0.051912685656726 w_k1_IBM_s103
                                 + 0.0157453865244907 w_k1_IBM_s104
                                 + 0.0282295895468064 w_k1_IBM_s105
                                 + 0.0183784541155274 w_k1_IBM_s106
                                 + 0.017458160654167 w_k1_IBM_s107
                                 + 0.0806342657547485 w_k1_IBM_s110
                                 + 0.0232831026411597 w_k1_IBM_s111
                                 + 0.019273354745227 w_k1_IBM_s112
                                 + 0.0282900617883231 w_k1_IBM_s113
                                 + 0.0501672070404084 w_k1_IBM_s114
                                 + 0.0686538442723761 w_k1_IBM_s115
                                 + 0.0855657422612066 w_k1_IBM_s116
                                 + 0.0434675989014378 w_k1_IBM_s117
                                 + 0.0435176529372085 w_k1_INTC_s000
                                 + 0.0156564902201754 w_k1_INTC_s001
                                 + 0.0421845081132637 w_k1_INTC_s002
                                 + 0.0535103068390603 w_k1_INTC_s003
                                 + 0.0192009877755172 w_k1_INTC_s004
                                 + 0.0203913803624194 w_k1_INTC_s005
                                 + 0.0210403920489576 w_k1_INTC_s006
                                 + 0.0340694515122512 w_k1_INTC_s007
                                 + 0.0196876434529869 w_k1_INTC_s010
                                 + 0.0137845307323172 w_k1_INTC_s011
                                 + 0.0133255765826732 w_k1_INTC_s012
                                 + 0.0270831079899968 w_k1_INTC_s013
                                 + 0.00516284605077552 w_k1_INTC_s014
                                 + 0.0181004555208811 w_k1_INTC_s015
                                 + 0.0111847915852925 w_k1_INTC_s016
                                 + 0.00569626764139951 w_k1_INTC_s017
                                 + 0.035770790337155 w_k1_INTC_s100
                                 + 0.0336734086597504 w_k1_INTC_s101
                                 + 0.0590517974032221 w_k1_INTC_s102
                                 + 0.0494498813341388 w_k1_INTC_s103
                                 + 0.0183147505249776 w_k1_INTC_s104
                                 + 0.0248497785150656 w_k1_INTC_s105
                                 + 0.0180576582262747 w_k1_INTC_s106
                                 + 0.0158906606863823 w_k1_INTC_s107
                                 + 0.0948223940493952 w_k1_INTC_s110
                                 + 0.0210227553848006 w_k1_INTC_s111
                                 + 0.0161357086198863 w_k1_INTC_s112
                                 + 0.0259487543458593 w_k1_INTC_s113
                                 + 0.0522510137262565 w_k1_INTC_s114
                                 + 0.0730449701800085 w_k1_INTC_s115
                                 + 0.0805785170343742 w_k1_INTC_s116
                                 + 0.0428993899657921 w_k1_INTC_s117
                                 + 0.0423008859661875 w_k1_KO_s000
                                 + 0.0162535914870089 w_k1_KO_s001
                                 + 0.0445738851277115 w_k1_KO_s002
                                 + 0.0532467896055641 w_k1_KO_s003
                                 + 0.0213862688631911 w_k1_KO_s004
                                 + 0.0214728723116027 w_k1_KO_s005
                                 + 0.0200955341421203 w_k1_KO_s006
                                 + 0.0301384931335242 w_k1_KO_s007
                                 + 0.0233979646468699 w_k1_KO_s010
                                 + 0.0162381996492919 w_k1_KO_s011
                                 + 0.0148905375986096 w_k1_KO_s012
                                 + 0.026234343168389 w_k1_KO_s013
                                 + 0.00582123715566459 w_k1_KO_s014
                                 + 0.0189721981832899 w_k1_KO_s015
                                 + 0.0116715562797536 w_k1_KO_s016
                                 + 0.00659338429782066 w_k1_KO_s017
                                 + 0.0375002125976364 w_k1_KO_s100
                                 + 0.0339529384524093 w_k1_KO_s101
                                 + 0.0550039295654992 w_k1_KO_s102
                                 + 0.0469202572622164 w_k1_KO_s103
                                 + 0.0170413916952946 w_k1_KO_s104
                                 + 0.0221098681132906 w_k1_KO_s105
                                 + 0.0185291683838683 w_k1_KO_s106
                                 + 0.0173189437550083 w_k1_KO_s107
                                 + 0.0759867581568208 w_k1_KO_s110
                                 + 0.0222368390382838 w_k1_KO_s111
                                 + 0.0179505875874513 w_k1_KO_s112
                                 + 0.0297576195314514 w_k1_KO_s113
                                 + 0.0478974414898099 w_k1_KO_s114
                                 + 0.0702172726508537 w_k1_KO_s115
                                 + 0.0866152485943365 w_k1_KO_s116
                                 + 0.0433283058088427 w_k1_KO_s117
                                 + 0.0592634896826394 w_k1_MAT_s000
                                 + 0.0149931261569656 w_k1_MAT_s001
                                 + 0.0527488419531598 w_k1_MAT_s002
                                 + 0.0451204295993089 w_k1_MAT_s003
                                 + 0.0248486434582707 w_k1_MAT_s004
                                 + 0.0227620262837127 w_k1_MAT_s005
                                 + 0.0278089660011301 w_k1_MAT_s006
                                 + 0.0337556904895367 w_k1_MAT_s007
                                 + 0.0270629620514833 w_k1_MAT_s010
                                 + 0.0199076012176555 w_k1_MAT_s011
                                 + 0.0125884158394156 w_k1_MAT_s012
                                 + 0.0252439910116208 w_k1_MAT_s013
                                 + 0.00672092696370034 w_k1_MAT_s014
                                 + 0.018003185419423 w_k1_MAT_s015
                                 + 0.012242439555302 w_k1_MAT_s016
                                 + 0.00715665143847401 w_k1_MAT_s017
                                 + 0.0278505410898207 w_k1_MAT_s100
                                 + 0.0363262029774573 w_k1_MAT_s101
                                 + 0.0504205967868693 w_k1_MAT_s102
                                 + 0.0431033952207203 w_k1_MAT_s103
                                 + 0.0153729940767644 w_k1_MAT_s104
                                 + 0.0296864539871154 w_k1_MAT_s105
                                 + 0.0207098120956151 w_k1_MAT_s106
                                 + 0.0182407178499418 w_k1_MAT_s107
                                 + 0.068012518855938 w_k1_MAT_s110
                                 + 0.0220097596083255 w_k1_MAT_s111
                                 + 0.0200843601294167 w_k1_MAT_s112
                                 + 0.0278250192597566 w_k1_MAT_s113
                                 + 0.0497431668305084 w_k1_MAT_s114
                                 + 0.0821787751060524 w_k1_MAT_s115
                                 + 0.100182491692707 w_k1_MAT_s116
                                 + 0.0381230553867094 w_k1_MAT_s117
                                 + 0.0467211933910147 w_k1_MSFT_s000
                                 + 0.0166669376493136 w_k1_MSFT_s001
                                 + 0.045569438273406 w_k1_MSFT_s002
                                 + 0.0543772231845591 w_k1_MSFT_s003
                                 + 0.022380602240849 w_k1_MSFT_s004
                                 + 0.0220918200138291 w_k1_MSFT_s005
                                 + 0.0220099324688807 w_k1_MSFT_s006
                                 + 0.028386470108605 w_k1_MSFT_s007
                                 + 0.0213548471050553 w_k1_MSFT_s010
                                 + 0.0164466016514998 w_k1_MSFT_s011
                                 + 0.0136457187225921 w_k1_MSFT_s012
                                 + 0.0267126112163918 w_k1_MSFT_s013
                                 + 0.00526434285399425 w_k1_MSFT_s014
                                 + 0.0183456833970538 w_k1_MSFT_s015
                                 + 0.0104755980431966 w_k1_MSFT_s016
                                 + 0.00681224740799168 w_k1_MSFT_s017
                                 + 0.033698649525719 w_k1_MSFT_s100
                                 + 0.0324714544509334 w_k1_MSFT_s101
                                 + 0.0564947233851632 w_k1_MSFT_s102
                                 + 0.0475484890813375 w_k1_MSFT_s103
                                 + 0.0167359422682227 w_k1_MSFT_s104
                                 + 0.0254684598605484 w_k1_MSFT_s105
                                 + 0.0205072468771972 w_k1_MSFT_s106
                                 + 0.0152668757846304 w_k1_MSFT_s107
                                 + 0.0813122215843518 w_k1_MSFT_s110
                                 + 0.0220742055861216 w_k1_MSFT_s111
                                 + 0.0181267294077026 w_k1_MSFT_s112
                                 + 0.0273564293417097 w_k1_MSFT_s113
                                 + 0.045419230640592 w_k1_MSFT_s114
                                 + 0.0665890452196717 w_k1_MSFT_s115
                                 + 0.0793941045909462 w_k1_MSFT_s116
                                 + 0.0406861589670237 w_k1_MSFT_s117
                                 + 0.0461175971278608 w_k1_PG_s000
                                 + 0.0160316084177415 w_k1_PG_s001
                                 + 0.0435227835960345 w_k1_PG_s002
                                 + 0.047739469268453 w_k1_PG_s003
                                 + 0.0241115423679082 w_k1_PG_s004
                                 + 0.0233960103665889 w_k1_PG_s005
                                 + 0.0225711708266041 w_k1_PG_s006
                                 + 0.0293520840111365 w_k1_PG_s007
                                 + 0.0229934820738312 w_k1_PG_s010
                                 + 0.0151450038520913 w_k1_PG_s011
                                 + 0.0152415907190524 w_k1_PG_s012
                                 + 0.0262551076432456 w_k1_PG_s013
                                 + 0.00529707551377458 w_k1_PG_s014
                                 + 0.0198102196225406 w_k1_PG_s015
                                 + 0.0110063853140349 w_k1_PG_s016
                                 + 0.00728414201424665 w_k1_PG_s017
                                 + 0.0306518588308335 w_k1_PG_s100
                                 + 0.0315520102666463 w_k1_PG_s101
                                 + 0.0519751727436326 w_k1_PG_s102
                                 + 0.0483462053957407 w_k1_PG_s103
                                 + 0.0145299539547296 w_k1_PG_s104
                                 + 0.0269153287459563 w_k1_PG_s105
                                 + 0.0181268056254845 w_k1_PG_s106
                                 + 0.0162473065276571 w_k1_PG_s107
                                 + 0.0808989894601895 w_k1_PG_s110
                                 + 0.0229201517420976 w_k1_PG_s111
                                 + 0.0177075765902121 w_k1_PG_s112
                                 + 0.0275669031739937 w_k1_PG_s113
                                 + 0.0487814924750917 w_k1_PG_s114
                                 + 0.069156128696461 w_k1_PG_s115
                                 + 0.0833009025697372 w_k1_PG_s116
                                 + 0.0444046386876941 w_k1_PG_s117 >= 1.2
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
