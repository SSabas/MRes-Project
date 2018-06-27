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
 w_k1_ADM_s0_return_constraint:  1.12381 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.85757 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.904112 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.04622 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.917578 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.16243 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.09841 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.46288 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.920211 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.934524 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.809913 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.735544 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.98865 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.930914 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.07142 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.825785 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.137 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.39884 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.29031 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.789863 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.699992 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.957734 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.13162 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.02572 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.19401 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.889011 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.728966 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.20816 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.869597 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.36805 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.972182 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.17875 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0095 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.0985 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.889604 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.730398 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.753423 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.12893 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.14239 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.8146 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.765935 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.11858 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.890882 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.22138 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.16883 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.80651 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.1345 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.831426 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.18137 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.676829 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.995402 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.19359 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.13271 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.948008 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.39604 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.74714 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.36784 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.752977 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.73648 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.782078 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.00785 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.895477 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.38378 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.984471 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.814392 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.52315 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.699834 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.32818 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.982553 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.28193 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.941986 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.07 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.918297 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.679117 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.653706 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.2519 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.18909 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.784668 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.745864 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.21244 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.859741 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.25329 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.13121 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.81918 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.870157 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.893162 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.05127 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.663093 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.938755 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.49539 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.12687 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.947781 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.14193 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.76446 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.33035 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.690321 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.554033 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.822197 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.00825 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.1075 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.81047 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.899679 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.802551 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.27605 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.695404 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.35562 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.850217 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.24733 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.01839 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.13011 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.901158 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.604916 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.648003 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.56224 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.11048 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.845789 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.855305 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.12948 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.907665 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.13222 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.24598 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.52244 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.944169 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.968122 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.871294 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.742909 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.944112 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.14555 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.09916 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.843232 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.17583 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.49426 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.14405 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.756433 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.645561 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.96321 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.20593 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.0486 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.45617 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.848064 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.848318 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.32599 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.814262 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.2727 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.892012 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0037 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.02043 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.16443 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.77187 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.734265 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.682943 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.41643 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13074 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.882696 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.793622 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.17246 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.812694 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.30826 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.763753 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.10044 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.14578 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.634777 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.55813 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.731972 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.907938 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.15548 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.833568 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.26315 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.07853 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.0301 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.67753 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.802806 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.00288 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.555702 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.765913 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.0958 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.52718 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.40037 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.966294 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.39373 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.697991 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.939164 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.05614 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.61942 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.81554 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.649235 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.62474 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.705683 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.11803 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.14017 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.00633 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.870187 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.89657 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.12942 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.952682 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.0483 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.22968 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.39456 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.860813 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06608 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.13638 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.727605 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.977237 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.30462 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.12135 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00594 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.32791 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.4515 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.27492 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.821426 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.836969 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.862033 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01865 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.06338 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.21872 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.85678 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.867909 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.36256 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.829621 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.14579 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.902084 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.23389 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.03531 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.10743 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.887443 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.963709 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.805826 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.46018 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1603 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.849374 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.770729 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.1045 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.898095 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.29821 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.0157 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.43741 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.02379 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.841815 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.21336 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.782627 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.987198 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.28391 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.945842 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.03043 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.10677 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.2925 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.45868 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.805999 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.782744 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.755785 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.824042 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.04003 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.47775 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.16239 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.860881 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.2075 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.738041 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.13849 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.786091 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.26917 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.870645 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.817768 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.04951 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.726059 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.900154 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.15869 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14966 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.776672 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.741596 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.1949 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.06271 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.61896 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.781859 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.28916 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.22446 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.609217 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.71403 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.750142 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.07079 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.09558 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.07731 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.21576 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23315 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.21762 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.70532 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.738839 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.978473 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.585048 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.58949 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.77013 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.24689 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.17105 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.810406 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.25251 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.6095 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.837106 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.02338 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.51563 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.735648 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.721229 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.29583 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.613722 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.833482 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.15087 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.10968 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.88083 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.829531 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.141 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.923077 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.02328 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.1832 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.38208 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.908535 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.04217 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.01668 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.803399 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.01583 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.29467 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.03469 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.951775 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.0897 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.27327 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.10958 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.84879 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.700051 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.931586 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.08473 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.977551 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.34335 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.873055 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.887297 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.10584 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.859994 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.22424 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.992923 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.22981 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.05798 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.24844 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.925095 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.847691 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.751805 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.45481 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11046 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.880049 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.797899 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.02838 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.953662 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.07882 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.05515 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.39946 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.08393 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.942204 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.08357 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.872349 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.967576 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.20551 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.02562 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.949501 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.1199 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.24264 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.21709 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.818705 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.81636 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.884987 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.967384 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.02517 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.27062 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.94478 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.867901 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.21015 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.85191 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.08706 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.989651 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.18927 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.08424 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.07669 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.04739 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.822506 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.912255 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.28023 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.314999771206561 z_k1_s0000
                                 - 0.457998670897439 z_k1_s0001
                                 - 0.244000068720251 z_k1_s0002
                                 - 0.260999791428298 z_k1_s0003
                                 - 0.312999784270325 z_k1_s0004
                                 - 0.24299990900218 z_k1_s0005
                                 - 0.425001380198852 z_k1_s0006
                                 - 1.06600134951113 z_k1_s0007
                                 - 0.206999414691058 z_k1_s0010
                                 - 0.179999962774867 z_k1_s0011
                                 - 0.0860000511035523 z_k1_s0012
                                 - 0.205999664620104 z_k1_s0013
                                 - 0.322000122852831 z_k1_s0014
                                 - 0.199999614194272 z_k1_s0015
                                 - 0.191000313555578 z_k1_s0016
                                 - 0.157999261213444 z_k1_s0017
                                 - 0.598999833754816 z_k1_s0020
                                 - 0.285999894275609 z_k1_s0021
                                 - 0.352999621177433 z_k1_s0022
                                 - 0.611999111549726 z_k1_s0023
                                 - 0.396000902119251 z_k1_s0024
                                 - 0.384999581755844 z_k1_s0025
                                 - 0.666998539372391 z_k1_s0026
                                 - 0.28899970002061 z_k1_s0027
                                 - 0.184000124631314 z_k1_s0030
                                 - 0.151999974717239 z_k1_s0031
                                 - 0.22499988838082 z_k1_s0032
                                 - 0.33300011240347 z_k1_s0033
                                 - 0.406999559048985 z_k1_s0034
                                 - 0.0819998744988317 z_k1_s0035
                                 - 0.159999882070825 z_k1_s0036
                                 - 0.192000205484811 z_k1_s0037
                                 - 0.511999498817006 z_k1_s0040
                                 - 1.19299535596222 z_k1_s0041
                                 - 0.401999142994073 z_k1_s0042
                                 - 0.983999036788934 z_k1_s0043
                                 - 0.387999156698296 z_k1_s0044
                                 - 0.320999454032273 z_k1_s0045
                                 - 0.432998880899526 z_k1_s0046
                                 - 0.401999142994073 z_k1_s0047
                                 - 0.373000375745318 z_k1_s0050
                                 - 0.47399914249742 z_k1_s0051
                                 - 0.420000720398658 z_k1_s0052
                                 - 1.01800063448128 z_k1_s0053
                                 - 0.332000479048328 z_k1_s0054
                                 - 0.805003286558605 z_k1_s0055
                                 - 0.639000673780899 z_k1_s0056
                                 - 0.254000265124172 z_k1_s0057
                                 - 0.237999126659086 z_k1_s0060
                                 - 0.568996220775584 z_k1_s0061
                                 - 0.992996223631462 z_k1_s0062
                                 - 0.491999644730299 z_k1_s0063
                                 - 0.443996990888373 z_k1_s0064
                                 - 0.225999033346524 z_k1_s0065
                                 - 0.590996328498845 z_k1_s0066
                                 - 0.247999141069786 z_k1_s0067
                                 - 0.498998740011795 z_k1_s0070
                                 - 0.203999591192015 z_k1_s0071
                                 - 0.390998332411205 z_k1_s0072
                                 - 0.474999322988292 z_k1_s0073
                                 - 0.274999432567205 z_k1_s0074
                                 - 0.337998356903125 z_k1_s0075
                                 - 0.603998462735741 z_k1_s0076
                                 - 0.244999555088861 z_k1_s0077
                                 - 0.625000063428687 z_k1_s0100
                                 - 1.62300359664031 z_k1_s0101
                                 - 2.2269964136005 z_k1_s0102
                                 - 1.01499671673708 z_k1_s0103
                                 - 0.906999399574293 z_k1_s0104
                                 - 0.850999416719876 z_k1_s0105
                                 - 1.5039989959759 z_k1_s0106
                                 - 0.760000039081293 z_k1_s0107
                                 - 1.02800091158074 z_k1_s0110
                                 - 0.603001649171334 z_k1_s0111
                                 - 0.733999949178666 z_k1_s0112
                                 - 0.352000250222148 z_k1_s0113
                                 - 0.570000133961778 z_k1_s0114
                                 - 0.629001878608371 z_k1_s0115
                                 - 0.463000332141067 z_k1_s0116
                                 - 0.925002450659112 z_k1_s0117
                                 - 0.160000027081598 z_k1_s0120
                                 - 0.220001135537894 z_k1_s0121
                                 - 0.499000892496246 z_k1_s0122
                                 - 0.234000808417325 z_k1_s0123
                                 - 0.152000004807505 z_k1_s0124
                                 - 0.411000856681354 z_k1_s0125
                                 - 0.261000177541941 z_k1_s0126
                                 - 0.155000143910373 z_k1_s0127
                                 - 0.287999814370959 z_k1_s0130
                                 - 0.475998986274399 z_k1_s0131
                                 - 0.352999104695126 z_k1_s0132
                                 - 0.279000022996817 z_k1_s0133
                                 - 0.425999006640555 z_k1_s0134
                                 - 0.177999859176502 z_k1_s0135
                                 - 0.683999174831123 z_k1_s0136
                                 - 0.732000794959208 z_k1_s0137
                                 - 0.752003056452329 z_k1_s0140
                                 - 0.620000696519354 z_k1_s0141
                                 - 0.99800083950012 z_k1_s0142
                                 - 0.326000585312091 z_k1_s0143
                                 - 1.40500030044935 z_k1_s0144
                                 - 0.759001442097892 z_k1_s0145
                                 - 0.718000733588441 z_k1_s0146
                                 - 0.889998026644652 z_k1_s0147
                                 - 0.460001114004275 z_k1_s0150
                                 - 0.847003329035194 z_k1_s0151
                                 - 0.597000468631713 z_k1_s0152
                                 - 0.36700164043713 z_k1_s0153
                                 - 0.384001869520718 z_k1_s0154
                                 - 0.601002658119536 z_k1_s0155
                                 - 0.827005347652736 z_k1_s0156
                                 - 0.239001161330853 z_k1_s0157
                                 - 0.596999414285127 z_k1_s0160
                                 - 0.596999414285127 z_k1_s0161
                                 - 0.166000006159072 z_k1_s0162
                                 - 0.688999030074358 z_k1_s0163
                                 - 0.688000941729482 z_k1_s0164
                                 - 0.199000170361835 z_k1_s0165
                                 - 0.212000179653814 z_k1_s0166
                                 - 0.50800137553305 z_k1_s0167
                                 - 0.709000951998838 z_k1_s0170
                                 - 0.382999334065539 z_k1_s0171
                                 - 0.517998859119505 z_k1_s0172
                                 - 0.531998482053297 z_k1_s0173
                                 - 1.67899395936557 z_k1_s0174
                                 - 0.429998983241069 z_k1_s0175
                                 - 0.924998794568544 z_k1_s0176
                                 - 0.646997797103646 z_k1_s0177
                                 - 0.149000140482253 z_k1_s1000
                                 - 0.121000058443912 z_k1_s1001
                                 - 0.252000131744358 z_k1_s1002
                                 - 0.168000290029808 z_k1_s1003
                                 - 0.295000033867263 z_k1_s1004
                                 - 0.218000162112239 z_k1_s1005
                                 - 0.25900030390412 z_k1_s1006
                                 - 0.560001640766812 z_k1_s1007
                                 - 0.155000181848359 z_k1_s1010
                                 - 0.154000085460895 z_k1_s1011
                                 - 0.36800048816646 z_k1_s1012
                                 - 0.135999990287493 z_k1_s1013
                                 - 0.257000538964193 z_k1_s1014
                                 - 0.232000862279181 z_k1_s1015
                                 - 0.125000023226024 z_k1_s1016
                                 - 0.395000722026616 z_k1_s1017
                                 - 0.135999539794345 z_k1_s1020
                                 - 0.207000036628634 z_k1_s1021
                                 - 0.109000119964901 z_k1_s1022
                                 - 0.184000618781386 z_k1_s1023
                                 - 0.198000493818989 z_k1_s1024
                                 - 0.206000233983212 z_k1_s1025
                                 - 0.0700000346884413 z_k1_s1026
                                 - 0.208999641919477 z_k1_s1027
                                 - 0.335001208480694 z_k1_s1030
                                 - 0.392000603187291 z_k1_s1031
                                 - 0.29600068646854 z_k1_s1032
                                 - 0.357002166335388 z_k1_s1033
                                 - 0.602000926323339 z_k1_s1034
                                 - 0.189000614223265 z_k1_s1035
                                 - 0.327999580705387 z_k1_s1036
                                 - 0.735002747980275 z_k1_s1037
                                 - 0.336002474910516 z_k1_s1040
                                 - 0.282001352333107 z_k1_s1041
                                 - 0.546002330473745 z_k1_s1042
                                 - 0.22500088899278 z_k1_s1043
                                 - 0.292000056879285 z_k1_s1044
                                 - 0.467000387516867 z_k1_s1045
                                 - 0.124000443029637 z_k1_s1046
                                 - 0.434000603500456 z_k1_s1047
                                 - 0.207999595281638 z_k1_s1050
                                 - 0.386000358360385 z_k1_s1051
                                 - 0.335000420613405 z_k1_s1052
                                 - 0.343000109200843 z_k1_s1053
                                 - 0.244000116927173 z_k1_s1054
                                 - 0.209999517428498 z_k1_s1055
                                 - 0.0780000405308208 z_k1_s1056
                                 - 0.119000175243297 z_k1_s1057
                                 - 0.439001563219644 z_k1_s1060
                                 - 0.335999636364826 z_k1_s1061
                                 - 0.43999843140721 z_k1_s1062
                                 - 0.504999242432425 z_k1_s1063
                                 - 0.281999828873941 z_k1_s1064
                                 - 0.407999796785666 z_k1_s1065
                                 - 0.46200043847587 z_k1_s1066
                                 - 1.29899849733374 z_k1_s1067
                                 - 0.34700016964321 z_k1_s1070
                                 - 0.22899953478943 z_k1_s1071
                                 - 0.191999697944657 z_k1_s1072
                                 - 0.231999406041961 z_k1_s1073
                                 - 0.731997894353589 z_k1_s1074
                                 - 0.766997409964229 z_k1_s1075
                                 - 0.304999224413773 z_k1_s1076
                                 - 0.248999388838081 z_k1_s1077
                                 - 0.504001634531496 z_k1_s1100
                                 - 0.298000362618949 z_k1_s1101
                                 - 0.199000267289871 z_k1_s1102
                                 - 0.165000236915241 z_k1_s1103
                                 - 0.406000211232122 z_k1_s1104
                                 - 0.340000303746294 z_k1_s1105
                                 - 0.261000804650179 z_k1_s1106
                                 - 0.168000232710052 z_k1_s1107
                                 - 0.119999864006793 z_k1_s1110
                                 - 0.121999878953554 z_k1_s1111
                                 - 0.328000127471339 z_k1_s1112
                                 - 0.207999230665709 z_k1_s1113
                                 - 0.141000279147488 z_k1_s1114
                                 - 0.117999849060033 z_k1_s1115
                                 - 0.125999908847075 z_k1_s1116
                                 - 0.127999794693981 z_k1_s1117
                                 - 0.240999297614114 z_k1_s1120
                                 - 0.169999568272852 z_k1_s1121
                                 - 0.176999534720601 z_k1_s1122
                                 - 0.438999706875293 z_k1_s1123
                                 - 0.52999780790003 z_k1_s1124
                                 - 0.48099901796312 z_k1_s1125
                                 - 0.144999653273844 z_k1_s1126
                                 - 0.254999474308944 z_k1_s1127
                                 - 0.0779999409496296 z_k1_s1130
                                 - 0.172999925434399 z_k1_s1131
                                 - 0.136999574442504 z_k1_s1132
                                 - 0.266999907358277 z_k1_s1133
                                 - 0.274000442884181 z_k1_s1134
                                 - 0.114999906302663 z_k1_s1135
                                 - 0.0859999614367541 z_k1_s1136
                                 - 0.164000161186218 z_k1_s1137
                                 - 0.0509999414306835 z_k1_s1140
                                 - 0.184000201424131 z_k1_s1141
                                 - 0.0559999370552593 z_k1_s1142
                                 - 0.0549999518703342 z_k1_s1143
                                 - 0.0889998663574897 z_k1_s1144
                                 - 0.131999856608822 z_k1_s1145
                                 - 0.0779998202234627 z_k1_s1146
                                 - 0.0519999963155588 z_k1_s1147
                                 - 0.306000926927299 z_k1_s1150
                                 - 0.20299981907393 z_k1_s1151
                                 - 0.151000455179152 z_k1_s1152
                                 - 0.0990002397822087 z_k1_s1153
                                 - 0.169000359439199 z_k1_s1154
                                 - 0.225000932006004 z_k1_s1155
                                 - 0.180000745604803 z_k1_s1156
                                 - 0.370001532632095 z_k1_s1157
                                 - 0.140000229016702 z_k1_s1160
                                 - 0.112999919477989 z_k1_s1161
                                 - 0.238999777293312 z_k1_s1162
                                 - 0.143999870674627 z_k1_s1163
                                 - 0.0999998650407512 z_k1_s1164
                                 - 0.265000350567399 z_k1_s1165
                                 - 0.0919998851254834 z_k1_s1166
                                 - 0.0679999453796798 z_k1_s1167
                                 - 0.182999986716113 z_k1_s1170
                                 - 0.100000114624693 z_k1_s1171
                                 - 0.178000127429928 z_k1_s1172
                                 - 0.301998980848112 z_k1_s1173
                                 - 0.344999753349974 z_k1_s1174
                                 - 0.368000638106943 z_k1_s1175
                                 - 0.360000232408834 z_k1_s1176
                                 - 0.417001151632197 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.06316 w_k1_ADM_s000
                                 + 0.946489 w_k1_AXP_s000
                                 + 1.23464 w_k1_DIS_s000
                                 + 1.00709 w_k1_FDX_s000
                                 + 1.37673 w_k1_IBM_s000
                                 + 1.03368 w_k1_INTC_s000
                                 + 1.14024 w_k1_KO_s000 + 1.32266 w_k1_MAT_s000
                                 + 1.08537 w_k1_MSFT_s000
                                 + 1.21204 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.935818 w_k1_ADM_s000
                                 + 1.13277 w_k1_AXP_s000
                                 + 0.891405 w_k1_DIS_s000
                                 + 0.856595 w_k1_FDX_s000
                                 + 1.06316 w_k1_IBM_s000
                                 + 0.912343 w_k1_INTC_s000
                                 + 1.01094 w_k1_KO_s000 + 1.53952 w_k1_MAT_s000
                                 + 0.908197 w_k1_MSFT_s000
                                 + 0.985957 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.41653 w_k1_ADM_s000
                                 + 1.75253 w_k1_AXP_s000
                                 + 1.68922 w_k1_DIS_s000
                                 + 1.36252 w_k1_FDX_s000
                                 + 1.22876 w_k1_IBM_s000
                                 + 1.32348 w_k1_INTC_s000
                                 + 1.31088 w_k1_KO_s000 + 1.5656 w_k1_MAT_s000
                                 + 1.26509 w_k1_MSFT_s000
                                 + 1.40606 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.09218 w_k1_ADM_s000
                                 + 1.28536 w_k1_AXP_s000
                                 + 1.56925 w_k1_DIS_s000
                                 + 1.24819 w_k1_FDX_s000
                                 + 0.954827 w_k1_IBM_s000
                                 + 1.14019 w_k1_INTC_s000
                                 + 1.13379 w_k1_KO_s000 + 1.02674 w_k1_MAT_s000
                                 + 1.23075 w_k1_MSFT_s000
                                 + 1.19258 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.82948 w_k1_ADM_s000
                                 + 0.607057 w_k1_AXP_s000
                                 + 0.669735 w_k1_DIS_s000
                                 + 0.744115 w_k1_FDX_s000
                                 + 1.00416 w_k1_IBM_s000
                                 + 0.743796 w_k1_INTC_s000
                                 + 0.841802 w_k1_KO_s000
                                 + 0.968137 w_k1_MAT_s000
                                 + 0.786345 w_k1_MSFT_s000
                                 + 0.950331 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.26912 w_k1_ADM_s000
                                 + 1.5774 w_k1_AXP_s000 + 1.22833 w_k1_DIS_s000
                                 + 1.0801 w_k1_FDX_s000 + 1.64628 w_k1_IBM_s000
                                 + 1.17456 w_k1_INTC_s000
                                 + 1.26021 w_k1_KO_s000 + 2.04127 w_k1_MAT_s000
                                 + 1.18973 w_k1_MSFT_s000
                                 + 1.23727 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.794853 w_k1_ADM_s000
                                 + 0.731328 w_k1_AXP_s000
                                 + 0.794942 w_k1_DIS_s000
                                 + 0.843795 w_k1_FDX_s000
                                 + 0.684264 w_k1_IBM_s000
                                 + 0.889771 w_k1_INTC_s000
                                 + 0.650159 w_k1_KO_s000
                                 + 0.510392 w_k1_MAT_s000
                                 + 0.860601 w_k1_MSFT_s000
                                 + 0.855921 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.00947 w_k1_ADM_s000
                                 + 0.997662 w_k1_AXP_s000
                                 + 1.01695 w_k1_DIS_s000
                                 + 1.03154 w_k1_FDX_s000
                                 + 0.982814 w_k1_IBM_s000
                                 + 0.860976 w_k1_INTC_s000
                                 + 1.05579 w_k1_KO_s000 + 1.15987 w_k1_MAT_s000
                                 + 1.01747 w_k1_MSFT_s000
                                 + 0.994473 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.784804 w_k1_ADM_s001
                                 + 0.899869 w_k1_AXP_s001
                                 + 0.845605 w_k1_DIS_s001
                                 + 1.05926 w_k1_FDX_s001
                                 + 0.503921 w_k1_IBM_s001
                                 + 1.05375 w_k1_INTC_s001
                                 + 0.73348 w_k1_KO_s001
                                 + 0.812337 w_k1_MAT_s001
                                 + 1.08319 w_k1_MSFT_s001
                                 + 0.889053 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.761413 w_k1_ADM_s001
                                 + 0.889095 w_k1_AXP_s001
                                 + 0.779261 w_k1_DIS_s001
                                 + 0.905786 w_k1_FDX_s001
                                 + 1.02819 w_k1_IBM_s001
                                 + 0.955451 w_k1_INTC_s001
                                 + 0.95755 w_k1_KO_s001 + 1.09852 w_k1_MAT_s001
                                 + 0.960119 w_k1_MSFT_s001
                                 + 1.01529 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.766508 w_k1_ADM_s001
                                 + 0.814493 w_k1_AXP_s001
                                 + 0.772772 w_k1_DIS_s001
                                 + 0.765388 w_k1_FDX_s001
                                 + 1.26686 w_k1_IBM_s001
                                 + 0.902718 w_k1_INTC_s001
                                 + 1.08802 w_k1_KO_s001 + 1.21596 w_k1_MAT_s001
                                 + 0.881915 w_k1_MSFT_s001
                                 + 0.913557 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.04822 w_k1_ADM_s001
                                 + 1.15885 w_k1_AXP_s001
                                 + 0.905551 w_k1_DIS_s001
                                 + 1.11045 w_k1_FDX_s001
                                 + 0.881813 w_k1_IBM_s001
                                 + 0.987195 w_k1_INTC_s001
                                 + 0.832844 w_k1_KO_s001
                                 + 0.834053 w_k1_MAT_s001
                                 + 1.05371 w_k1_MSFT_s001
                                 + 0.876852 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.21243 w_k1_ADM_s001
                                 + 1.42641 w_k1_AXP_s001
                                 + 1.33396 w_k1_DIS_s001
                                 + 1.25751 w_k1_FDX_s001 + 1.4588 w_k1_IBM_s001
                                 + 1.20413 w_k1_INTC_s001
                                 + 1.12069 w_k1_KO_s001 + 1.53523 w_k1_MAT_s001
                                 + 1.20061 w_k1_MSFT_s001
                                 + 1.08499 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.722112 w_k1_ADM_s001
                                 + 0.802668 w_k1_AXP_s001
                                 + 1.16314 w_k1_DIS_s001
                                 + 0.997761 w_k1_FDX_s001
                                 + 0.833734 w_k1_IBM_s001
                                 + 0.999729 w_k1_INTC_s001
                                 + 0.907636 w_k1_KO_s001
                                 + 0.881876 w_k1_MAT_s001
                                 + 1.1258 w_k1_MSFT_s001
                                 + 0.877585 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.878158 w_k1_ADM_s001
                                 + 0.771635 w_k1_AXP_s001
                                 + 1.16227 w_k1_DIS_s001
                                 + 0.991035 w_k1_FDX_s001
                                 + 1.42486 w_k1_IBM_s001
                                 + 0.831144 w_k1_INTC_s001
                                 + 1.17623 w_k1_KO_s001 + 1.04474 w_k1_MAT_s001
                                 + 1.01332 w_k1_MSFT_s001
                                 + 1.02512 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.684715 w_k1_ADM_s001
                                 + 0.71446 w_k1_AXP_s001
                                 + 0.560344 w_k1_DIS_s001
                                 + 0.78985 w_k1_FDX_s001
                                 + 0.653653 w_k1_IBM_s001
                                 + 0.819776 w_k1_INTC_s001
                                 + 0.639044 w_k1_KO_s001 + 1.0443 w_k1_MAT_s001
                                 + 0.841196 w_k1_MSFT_s001
                                 + 0.776787 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.09852 w_k1_ADM_s002
                                 + 1.2495 w_k1_AXP_s002 + 1.24638 w_k1_DIS_s002
                                 + 1.05979 w_k1_FDX_s002
                                 + 1.41996 w_k1_IBM_s002
                                 + 1.36115 w_k1_INTC_s002
                                 + 1.23122 w_k1_KO_s002 + 1.20585 w_k1_MAT_s002
                                 + 1.18249 w_k1_MSFT_s002
                                 + 1.03156 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.988759 w_k1_ADM_s002
                                 + 1.06347 w_k1_AXP_s002
                                 + 1.08986 w_k1_DIS_s002
                                 + 1.05761 w_k1_FDX_s002
                                 + 0.877655 w_k1_IBM_s002
                                 + 1.1685 w_k1_INTC_s002 + 0.9867 w_k1_KO_s002
                                 + 0.91979 w_k1_MAT_s002 + 1.131 w_k1_MSFT_s002
                                 + 0.91046 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.965918 w_k1_ADM_s002
                                 + 0.792954 w_k1_AXP_s002
                                 + 0.861816 w_k1_DIS_s002
                                 + 0.878851 w_k1_FDX_s002
                                 + 0.651209 w_k1_IBM_s002
                                 + 1.08836 w_k1_INTC_s002
                                 + 0.724216 w_k1_KO_s002
                                 + 0.588789 w_k1_MAT_s002
                                 + 0.863084 w_k1_MSFT_s002
                                 + 0.849755 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.755885 w_k1_ADM_s002
                                 + 0.894768 w_k1_AXP_s002
                                 + 0.92874 w_k1_DIS_s002
                                 + 0.931684 w_k1_FDX_s002
                                 + 1.04195 w_k1_IBM_s002
                                 + 1.12572 w_k1_INTC_s002
                                 + 0.876821 w_k1_KO_s002
                                 + 0.890432 w_k1_MAT_s002
                                 + 1.048 w_k1_MSFT_s002 + 0.86242 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.972935 w_k1_ADM_s002
                                 + 1.42357 w_k1_AXP_s002 + 1.6465 w_k1_DIS_s002
                                 + 1.37239 w_k1_FDX_s002
                                 + 1.10509 w_k1_IBM_s002
                                 + 1.63025 w_k1_INTC_s002
                                 + 1.17008 w_k1_KO_s002 + 1.05905 w_k1_MAT_s002
                                 + 1.52003 w_k1_MSFT_s002
                                 + 1.12372 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.31404 w_k1_ADM_s002
                                 + 1.17936 w_k1_AXP_s002 + 1.2394 w_k1_DIS_s002
                                 + 1.13222 w_k1_FDX_s002
                                 + 0.822262 w_k1_IBM_s002
                                 + 1.24309 w_k1_INTC_s002
                                 + 0.836202 w_k1_KO_s002
                                 + 0.934412 w_k1_MAT_s002
                                 + 1.11319 w_k1_MSFT_s002
                                 + 0.97617 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.772951 w_k1_ADM_s002
                                 + 0.673914 w_k1_AXP_s002
                                 + 0.822384 w_k1_DIS_s002
                                 + 0.772366 w_k1_FDX_s002
                                 + 0.895899 w_k1_IBM_s002
                                 + 0.997551 w_k1_INTC_s002
                                 + 0.831768 w_k1_KO_s002
                                 + 0.697118 w_k1_MAT_s002
                                 + 0.886183 w_k1_MSFT_s002
                                 + 0.8279 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.07519 w_k1_ADM_s002
                                 + 1.42231 w_k1_AXP_s002
                                 + 1.00406 w_k1_DIS_s002
                                 + 1.17247 w_k1_FDX_s002
                                 + 1.08744 w_k1_IBM_s002
                                 + 1.17332 w_k1_INTC_s002
                                 + 1.00279 w_k1_KO_s002 + 1.70365 w_k1_MAT_s002
                                 + 1.0641 w_k1_MSFT_s002 + 1.09333 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.896892 w_k1_ADM_s003
                                 + 0.839814 w_k1_AXP_s003
                                 + 0.960322 w_k1_DIS_s003
                                 + 0.864912 w_k1_FDX_s003
                                 + 1.23866 w_k1_IBM_s003
                                 + 0.813417 w_k1_INTC_s003
                                 + 1.02215 w_k1_KO_s003 + 1.25166 w_k1_MAT_s003
                                 + 0.948286 w_k1_MSFT_s003
                                 + 1.10392 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.745809 w_k1_ADM_s003
                                 + 1.13224 w_k1_AXP_s003
                                 + 1.39201 w_k1_DIS_s003
                                 + 1.07303 w_k1_FDX_s003
                                 + 1.44328 w_k1_IBM_s003
                                 + 1.20686 w_k1_INTC_s003
                                 + 1.30576 w_k1_KO_s003 + 1.27706 w_k1_MAT_s003
                                 + 1.07162 w_k1_MSFT_s003 + 1.11 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.803357 w_k1_ADM_s003
                                 + 0.720124 w_k1_AXP_s003
                                 + 0.671525 w_k1_DIS_s003
                                 + 0.793307 w_k1_FDX_s003
                                 + 0.788845 w_k1_IBM_s003
                                 + 0.755244 w_k1_INTC_s003
                                 + 0.769698 w_k1_KO_s003
                                 + 0.870071 w_k1_MAT_s003
                                 + 0.73696 w_k1_MSFT_s003
                                 + 0.892061 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.772733 w_k1_ADM_s003
                                 + 0.848249 w_k1_AXP_s003
                                 + 0.89631 w_k1_DIS_s003
                                 + 0.958093 w_k1_FDX_s003
                                 + 0.747697 w_k1_IBM_s003
                                 + 0.887922 w_k1_INTC_s003
                                 + 0.887636 w_k1_KO_s003
                                 + 0.696776 w_k1_MAT_s003
                                 + 0.969987 w_k1_MSFT_s003
                                 + 0.930418 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.00982 w_k1_ADM_s003
                                 + 1.01919 w_k1_AXP_s003
                                 + 1.20928 w_k1_DIS_s003
                                 + 1.09124 w_k1_FDX_s003
                                 + 0.942855 w_k1_IBM_s003
                                 + 0.92694 w_k1_INTC_s003
                                 + 0.996756 w_k1_KO_s003
                                 + 0.914966 w_k1_MAT_s003
                                 + 1.00252 w_k1_MSFT_s003
                                 + 1.02587 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.817158 w_k1_ADM_s003
                                 + 0.803932 w_k1_AXP_s003
                                 + 0.634365 w_k1_DIS_s003
                                 + 0.945298 w_k1_FDX_s003
                                 + 0.530003 w_k1_IBM_s003
                                 + 0.84947 w_k1_INTC_s003
                                 + 0.654863 w_k1_KO_s003
                                 + 0.617034 w_k1_MAT_s003
                                 + 0.889226 w_k1_MSFT_s003
                                 + 0.870093 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.1479 w_k1_ADM_s003
                                 + 1.33648 w_k1_AXP_s003 + 1.6927 w_k1_DIS_s003
                                 + 1.35074 w_k1_FDX_s003
                                 + 1.29546 w_k1_IBM_s003
                                 + 1.22043 w_k1_INTC_s003
                                 + 1.35925 w_k1_KO_s003 + 1.32657 w_k1_MAT_s003
                                 + 1.39058 w_k1_MSFT_s003
                                 + 1.18474 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.00922 w_k1_ADM_s003
                                 + 1.18833 w_k1_AXP_s003
                                 + 0.952064 w_k1_DIS_s003
                                 + 1.16638 w_k1_FDX_s003
                                 + 0.85708 w_k1_IBM_s003
                                 + 0.924035 w_k1_INTC_s003
                                 + 0.724756 w_k1_KO_s003
                                 + 0.788622 w_k1_MAT_s003
                                 + 0.881207 w_k1_MSFT_s003
                                 + 0.99353 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.18386 w_k1_ADM_s004
                                 + 1.19894 w_k1_AXP_s004
                                 + 1.14504 w_k1_DIS_s004
                                 + 1.34628 w_k1_FDX_s004
                                 + 0.861846 w_k1_IBM_s004
                                 + 1.02282 w_k1_INTC_s004
                                 + 1.03158 w_k1_KO_s004
                                 + 0.694176 w_k1_MAT_s004
                                 + 1.16772 w_k1_MSFT_s004 + 1.0339 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.08034 w_k1_ADM_s004
                                 + 0.809897 w_k1_AXP_s004
                                 + 0.862828 w_k1_DIS_s004
                                 + 1.01932 w_k1_FDX_s004
                                 + 0.845019 w_k1_IBM_s004
                                 + 0.874313 w_k1_INTC_s004
                                 + 0.804091 w_k1_KO_s004
                                 + 0.754759 w_k1_MAT_s004
                                 + 0.924502 w_k1_MSFT_s004
                                 + 0.886485 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.83329 w_k1_ADM_s004
                                 + 0.717822 w_k1_AXP_s004
                                 + 0.68306 w_k1_DIS_s004
                                 + 0.762892 w_k1_FDX_s004
                                 + 1.06042 w_k1_IBM_s004
                                 + 0.817802 w_k1_INTC_s004
                                 + 0.968394 w_k1_KO_s004
                                 + 0.816899 w_k1_MAT_s004
                                 + 0.799089 w_k1_MSFT_s004
                                 + 0.898597 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10314 w_k1_ADM_s004
                                 + 1.17792 w_k1_AXP_s004
                                 + 1.16203 w_k1_DIS_s004
                                 + 1.12876 w_k1_FDX_s004
                                 + 1.23925 w_k1_IBM_s004
                                 + 0.894445 w_k1_INTC_s004
                                 + 1.14113 w_k1_KO_s004 + 1.30113 w_k1_MAT_s004
                                 + 1.03665 w_k1_MSFT_s004
                                 + 1.10391 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.65803 w_k1_ADM_s004
                                 + 1.5189 w_k1_AXP_s004 + 1.88721 w_k1_DIS_s004
                                 + 1.52458 w_k1_FDX_s004
                                 + 1.68278 w_k1_IBM_s004
                                 + 1.1946 w_k1_INTC_s004 + 1.57642 w_k1_KO_s004
                                 + 1.59091 w_k1_MAT_s004
                                 + 1.28957 w_k1_MSFT_s004
                                 + 1.24024 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.01435 w_k1_ADM_s004
                                 + 0.943486 w_k1_AXP_s004
                                 + 0.878412 w_k1_DIS_s004
                                 + 0.906645 w_k1_FDX_s004
                                 + 1.42285 w_k1_IBM_s004
                                 + 0.748131 w_k1_INTC_s004
                                 + 1.2466 w_k1_KO_s004 + 1.39543 w_k1_MAT_s004
                                 + 0.907962 w_k1_MSFT_s004
                                 + 1.06028 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.823046 w_k1_ADM_s004
                                 + 0.49419 w_k1_AXP_s004
                                 + 0.613487 w_k1_DIS_s004
                                 + 0.737707 w_k1_FDX_s004
                                 + 0.646217 w_k1_IBM_s004
                                 + 0.744993 w_k1_INTC_s004
                                 + 0.73075 w_k1_KO_s004 + 0.50472 w_k1_MAT_s004
                                 + 0.798458 w_k1_MSFT_s004
                                 + 0.795969 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.908141 w_k1_ADM_s004
                                 + 0.607072 w_k1_AXP_s004
                                 + 0.531926 w_k1_DIS_s004
                                 + 0.691531 w_k1_FDX_s004
                                 + 0.858361 w_k1_IBM_s004
                                 + 0.688856 w_k1_INTC_s004
                                 + 0.758949 w_k1_KO_s004
                                 + 0.824955 w_k1_MAT_s004
                                 + 0.712214 w_k1_MSFT_s004
                                 + 0.835539 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.992719 w_k1_ADM_s005
                                 + 1.37692 w_k1_AXP_s005
                                 + 1.04258 w_k1_DIS_s005
                                 + 1.04583 w_k1_FDX_s005
                                 + 1.75352 w_k1_IBM_s005
                                 + 1.24954 w_k1_INTC_s005
                                 + 1.39391 w_k1_KO_s005 + 1.68677 w_k1_MAT_s005
                                 + 1.10082 w_k1_MSFT_s005
                                 + 1.08922 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.09356 w_k1_ADM_s005
                                 + 1.10662 w_k1_AXP_s005
                                 + 1.12479 w_k1_DIS_s005
                                 + 1.08199 w_k1_FDX_s005
                                 + 1.38897 w_k1_IBM_s005
                                 + 1.39674 w_k1_INTC_s005
                                 + 1.18855 w_k1_KO_s005 + 1.06386 w_k1_MAT_s005
                                 + 1.05118 w_k1_MSFT_s005
                                 + 1.10518 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.724966 w_k1_ADM_s005
                                 + 0.837474 w_k1_AXP_s005
                                 + 0.665129 w_k1_DIS_s005
                                 + 0.769279 w_k1_FDX_s005
                                 + 1.24158 w_k1_IBM_s005
                                 + 0.986509 w_k1_INTC_s005
                                 + 0.863611 w_k1_KO_s005
                                 + 1.03496 w_k1_MAT_s005
                                 + 0.773872 w_k1_MSFT_s005
                                 + 0.834617 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.29274 w_k1_ADM_s005
                                 + 1.27028 w_k1_AXP_s005
                                 + 1.19347 w_k1_DIS_s005 + 1.2349 w_k1_FDX_s005
                                 + 0.861928 w_k1_IBM_s005
                                 + 1.37097 w_k1_INTC_s005
                                 + 0.932666 w_k1_KO_s005
                                 + 0.816347 w_k1_MAT_s005
                                 + 1.13105 w_k1_MSFT_s005
                                 + 1.10651 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.17753 w_k1_ADM_s005
                                 + 1.45156 w_k1_AXP_s005
                                 + 1.30001 w_k1_DIS_s005
                                 + 1.17054 w_k1_FDX_s005
                                 + 1.19417 w_k1_IBM_s005
                                 + 1.27015 w_k1_INTC_s005
                                 + 1.11407 w_k1_KO_s005 + 1.13495 w_k1_MAT_s005
                                 + 1.11041 w_k1_MSFT_s005
                                 + 1.21158 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.944734 w_k1_ADM_s005
                                 + 0.677809 w_k1_AXP_s005
                                 + 0.788113 w_k1_DIS_s005
                                 + 0.855102 w_k1_FDX_s005
                                 + 0.858604 w_k1_IBM_s005
                                 + 1.02251 w_k1_INTC_s005
                                 + 0.87859 w_k1_KO_s005
                                 + 0.668704 w_k1_MAT_s005
                                 + 1.02532 w_k1_MSFT_s005
                                 + 0.883027 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.957646 w_k1_ADM_s005
                                 + 0.791679 w_k1_AXP_s005
                                 + 0.682457 w_k1_DIS_s005
                                 + 0.910819 w_k1_FDX_s005
                                 + 0.689773 w_k1_IBM_s005
                                 + 0.967043 w_k1_INTC_s005
                                 + 0.748962 w_k1_KO_s005
                                 + 0.717286 w_k1_MAT_s005
                                 + 0.823304 w_k1_MSFT_s005
                                 + 0.852316 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.29942 w_k1_ADM_s005
                                 + 1.29154 w_k1_AXP_s005
                                 + 0.910799 w_k1_DIS_s005
                                 + 1.18639 w_k1_FDX_s005
                                 + 0.628593 w_k1_IBM_s005
                                 + 1.18533 w_k1_INTC_s005
                                 + 0.779218 w_k1_KO_s005
                                 + 0.771354 w_k1_MAT_s005
                                 + 0.974346 w_k1_MSFT_s005
                                 + 1.02981 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.677743 w_k1_ADM_s006
                                 + 0.588057 w_k1_AXP_s006
                                 + 0.734911 w_k1_DIS_s006
                                 + 0.709113 w_k1_FDX_s006
                                 + 0.962348 w_k1_IBM_s006
                                 + 0.753333 w_k1_INTC_s006
                                 + 0.864244 w_k1_KO_s006
                                 + 0.653512 w_k1_MAT_s006
                                 + 0.721703 w_k1_MSFT_s006
                                 + 0.886772 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.08959 w_k1_ADM_s006
                                 + 1.50207 w_k1_AXP_s006 + 1.4912 w_k1_DIS_s006
                                 + 1.24922 w_k1_FDX_s006
                                 + 1.24614 w_k1_IBM_s006
                                 + 1.45153 w_k1_INTC_s006 + 1.1668 w_k1_KO_s006
                                 + 1.14538 w_k1_MAT_s006
                                 + 1.14335 w_k1_MSFT_s006
                                 + 1.23407 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.896164 w_k1_ADM_s006
                                 + 1.03972 w_k1_AXP_s006
                                 + 0.83771 w_k1_DIS_s006
                                 + 0.962436 w_k1_FDX_s006
                                 + 0.937496 w_k1_IBM_s006
                                 + 0.839471 w_k1_INTC_s006
                                 + 0.854652 w_k1_KO_s006
                                 + 1.03519 w_k1_MAT_s006
                                 + 0.926017 w_k1_MSFT_s006 + 1.065 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.752745 w_k1_ADM_s006
                                 + 0.779405 w_k1_AXP_s006
                                 + 0.97377 w_k1_DIS_s006
                                 + 0.836058 w_k1_FDX_s006
                                 + 1.15653 w_k1_IBM_s006
                                 + 0.948509 w_k1_INTC_s006
                                 + 1.11698 w_k1_KO_s006 + 1.45626 w_k1_MAT_s006
                                 + 0.851247 w_k1_MSFT_s006
                                 + 1.04957 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06745 w_k1_ADM_s006
                                 + 1.41214 w_k1_AXP_s006
                                 + 1.04599 w_k1_DIS_s006
                                 + 1.24882 w_k1_FDX_s006
                                 + 0.814876 w_k1_IBM_s006
                                 + 1.35431 w_k1_INTC_s006
                                 + 0.883145 w_k1_KO_s006
                                 + 0.623677 w_k1_MAT_s006
                                 + 1.00982 w_k1_MSFT_s006
                                 + 1.10434 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.560932 w_k1_ADM_s006
                                 + 0.61328 w_k1_AXP_s006
                                 + 0.643863 w_k1_DIS_s006
                                 + 0.791443 w_k1_FDX_s006
                                 + 0.687768 w_k1_IBM_s006
                                 + 0.745412 w_k1_INTC_s006
                                 + 0.68038 w_k1_KO_s006
                                 + 0.761946 w_k1_MAT_s006
                                 + 0.751771 w_k1_MSFT_s006
                                 + 0.815172 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.921639 w_k1_ADM_s006
                                 + 1.11878 w_k1_AXP_s006
                                 + 1.33103 w_k1_DIS_s006 + 1.173 w_k1_FDX_s006
                                 + 1.04975 w_k1_IBM_s006
                                 + 1.20758 w_k1_INTC_s006
                                 + 1.08552 w_k1_KO_s006 + 0.78046 w_k1_MAT_s006
                                 + 1.10399 w_k1_MSFT_s006
                                 + 1.14815 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.842342 w_k1_ADM_s006
                                 + 0.87954 w_k1_AXP_s006
                                 + 0.80745 w_k1_DIS_s006
                                 + 1.02041 w_k1_FDX_s006
                                 + 0.638367 w_k1_IBM_s006
                                 + 1.20142 w_k1_INTC_s006
                                 + 0.701911 w_k1_KO_s006
                                 + 0.527545 w_k1_MAT_s006
                                 + 0.969803 w_k1_MSFT_s006
                                 + 0.927106 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.17344 w_k1_ADM_s007
                                 + 0.932791 w_k1_AXP_s007
                                 + 1.24207 w_k1_DIS_s007
                                 + 1.15125 w_k1_FDX_s007
                                 + 0.831538 w_k1_IBM_s007
                                 + 0.93406 w_k1_INTC_s007
                                 + 1.04729 w_k1_KO_s007
                                 + 0.828697 w_k1_MAT_s007
                                 + 1.03123 w_k1_MSFT_s007
                                 + 0.999558 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.29328 w_k1_ADM_s007
                                 + 1.56318 w_k1_AXP_s007
                                 + 1.62851 w_k1_DIS_s007
                                 + 1.34276 w_k1_FDX_s007
                                 + 1.58212 w_k1_IBM_s007
                                 + 1.20778 w_k1_INTC_s007
                                 + 1.30672 w_k1_KO_s007 + 1.77842 w_k1_MAT_s007
                                 + 1.20518 w_k1_MSFT_s007
                                 + 1.26402 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.991175 w_k1_ADM_s007
                                 + 1.07113 w_k1_AXP_s007
                                 + 0.903404 w_k1_DIS_s007
                                 + 0.986927 w_k1_FDX_s007
                                 + 1.32647 w_k1_IBM_s007
                                 + 0.938663 w_k1_INTC_s007
                                 + 1.09124 w_k1_KO_s007 + 1.46712 w_k1_MAT_s007
                                 + 0.904476 w_k1_MSFT_s007
                                 + 0.98766 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.20214 w_k1_ADM_s007
                                 + 1.28669 w_k1_AXP_s007
                                 + 1.01347 w_k1_DIS_s007
                                 + 1.00919 w_k1_FDX_s007
                                 + 1.02533 w_k1_IBM_s007
                                 + 0.887957 w_k1_INTC_s007
                                 + 1.01946 w_k1_KO_s007 + 1.35197 w_k1_MAT_s007
                                 + 0.967304 w_k1_MSFT_s007
                                 + 1.01618 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.789197 w_k1_ADM_s007
                                 + 0.594441 w_k1_AXP_s007
                                 + 0.710039 w_k1_DIS_s007
                                 + 0.785319 w_k1_FDX_s007
                                 + 0.853739 w_k1_IBM_s007
                                 + 0.893032 w_k1_INTC_s007
                                 + 0.796566 w_k1_KO_s007
                                 + 0.748271 w_k1_MAT_s007
                                 + 0.813203 w_k1_MSFT_s007
                                 + 0.888148 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.979424 w_k1_ADM_s007
                                 + 0.757686 w_k1_AXP_s007
                                 + 0.63593 w_k1_DIS_s007
                                 + 0.861846 w_k1_FDX_s007
                                 + 0.80776 w_k1_IBM_s007
                                 + 0.829913 w_k1_INTC_s007
                                 + 0.823723 w_k1_KO_s007
                                 + 0.875143 w_k1_MAT_s007
                                 + 0.788717 w_k1_MSFT_s007
                                 + 0.871139 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.97647 w_k1_ADM_s007
                                 + 0.8501 w_k1_AXP_s007
                                 + 0.851093 w_k1_DIS_s007
                                 + 0.938288 w_k1_FDX_s007
                                 + 0.746337 w_k1_IBM_s007
                                 + 0.9222 w_k1_INTC_s007
                                 + 0.875506 w_k1_KO_s007
                                 + 0.605587 w_k1_MAT_s007
                                 + 1.04217 w_k1_MSFT_s007
                                 + 0.89554 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.52791 w_k1_ADM_s007
                                 + 1.44795 w_k1_AXP_s007
                                 + 1.50557 w_k1_DIS_s007
                                 + 1.40008 w_k1_FDX_s007
                                 + 1.22765 w_k1_IBM_s007
                                 + 0.980805 w_k1_INTC_s007
                                 + 1.22077 w_k1_KO_s007 + 1.46919 w_k1_MAT_s007
                                 + 1.18061 w_k1_MSFT_s007
                                 + 1.08019 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.21762 w_k1_ADM_s010
                                 + 1.52767 w_k1_AXP_s010
                                 + 1.07036 w_k1_DIS_s010
                                 + 1.18088 w_k1_FDX_s010
                                 + 1.33689 w_k1_IBM_s010
                                 + 1.04919 w_k1_INTC_s010
                                 + 1.31552 w_k1_KO_s010 + 1.66112 w_k1_MAT_s010
                                 + 1.01079 w_k1_MSFT_s010
                                 + 1.25331 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.33188 w_k1_ADM_s010
                                 + 1.15767 w_k1_AXP_s010
                                 + 1.28562 w_k1_DIS_s010
                                 + 1.30065 w_k1_FDX_s010
                                 + 0.82417 w_k1_IBM_s010
                                 + 1.0498 w_k1_INTC_s010 + 1.14951 w_k1_KO_s010
                                 + 0.795241 w_k1_MAT_s010
                                 + 1.15204 w_k1_MSFT_s010
                                 + 1.11389 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.06689 w_k1_ADM_s010
                                 + 0.978695 w_k1_AXP_s010
                                 + 0.971677 w_k1_DIS_s010
                                 + 0.961052 w_k1_FDX_s010
                                 + 0.893951 w_k1_IBM_s010
                                 + 1.09698 w_k1_INTC_s010
                                 + 1.03107 w_k1_KO_s010
                                 + 0.786117 w_k1_MAT_s010
                                 + 1.01972 w_k1_MSFT_s010
                                 + 0.99673 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.15265 w_k1_ADM_s010
                                 + 0.989116 w_k1_AXP_s010
                                 + 1.12483 w_k1_DIS_s010
                                 + 1.01458 w_k1_FDX_s010
                                 + 1.54479 w_k1_IBM_s010
                                 + 1.15784 w_k1_INTC_s010
                                 + 1.35647 w_k1_KO_s010 + 1.04961 w_k1_MAT_s010
                                 + 0.989361 w_k1_MSFT_s010
                                 + 1.08386 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.02226 w_k1_ADM_s010
                                 + 0.86637 w_k1_AXP_s010
                                 + 0.766516 w_k1_DIS_s010
                                 + 0.85999 w_k1_FDX_s010
                                 + 1.00766 w_k1_IBM_s010
                                 + 0.775464 w_k1_INTC_s010
                                 + 0.884713 w_k1_KO_s010
                                 + 0.616382 w_k1_MAT_s010
                                 + 0.804543 w_k1_MSFT_s010
                                 + 0.89519 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.44466 w_k1_ADM_s010
                                 + 1.07516 w_k1_AXP_s010
                                 + 1.69786 w_k1_DIS_s010
                                 + 1.27386 w_k1_FDX_s010
                                 + 1.05926 w_k1_IBM_s010
                                 + 1.26275 w_k1_INTC_s010
                                 + 1.60554 w_k1_KO_s010
                                 + 0.664965 w_k1_MAT_s010
                                 + 1.47389 w_k1_MSFT_s010 + 1.2106 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.829116 w_k1_ADM_s010
                                 + 0.781768 w_k1_AXP_s010
                                 + 0.617771 w_k1_DIS_s010
                                 + 0.797677 w_k1_FDX_s010
                                 + 0.722932 w_k1_IBM_s010
                                 + 0.717649 w_k1_INTC_s010
                                 + 0.717005 w_k1_KO_s010
                                 + 0.804072 w_k1_MAT_s010
                                 + 0.791687 w_k1_MSFT_s010
                                 + 0.793833 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.844741 w_k1_ADM_s010
                                 + 0.942898 w_k1_AXP_s010
                                 + 0.665977 w_k1_DIS_s010
                                 + 0.762028 w_k1_FDX_s010
                                 + 1.25216 w_k1_IBM_s010
                                 + 0.932888 w_k1_INTC_s010
                                 + 0.974319 w_k1_KO_s010
                                 + 0.870605 w_k1_MAT_s010
                                 + 0.763446 w_k1_MSFT_s010
                                 + 0.895957 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.24375 w_k1_ADM_s011
                                 + 1.26951 w_k1_AXP_s011
                                 + 1.45918 w_k1_DIS_s011
                                 + 1.29733 w_k1_FDX_s011
                                 + 0.996716 w_k1_IBM_s011
                                 + 0.982386 w_k1_INTC_s011
                                 + 1.22227 w_k1_KO_s011 + 1.16056 w_k1_MAT_s011
                                 + 1.14587 w_k1_MSFT_s011
                                 + 1.25918 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.13895 w_k1_ADM_s011
                                 + 1.32592 w_k1_AXP_s011
                                 + 1.01426 w_k1_DIS_s011
                                 + 1.00405 w_k1_FDX_s011 + 1.4887 w_k1_IBM_s011
                                 + 1.05125 w_k1_INTC_s011
                                 + 1.23905 w_k1_KO_s011 + 1.72045 w_k1_MAT_s011
                                 + 0.955608 w_k1_MSFT_s011
                                 + 1.13832 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.894081 w_k1_ADM_s011
                                 + 0.830823 w_k1_AXP_s011
                                 + 0.9756 w_k1_DIS_s011
                                 + 0.982236 w_k1_FDX_s011
                                 + 0.72736 w_k1_IBM_s011
                                 + 0.998014 w_k1_INTC_s011
                                 + 0.944641 w_k1_KO_s011
                                 + 0.644105 w_k1_MAT_s011
                                 + 0.892177 w_k1_MSFT_s011
                                 + 0.923769 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.90576 w_k1_ADM_s011
                                 + 0.853054 w_k1_AXP_s011
                                 + 0.597271 w_k1_DIS_s011
                                 + 0.684617 w_k1_FDX_s011
                                 + 1.39941 w_k1_IBM_s011
                                 + 0.686909 w_k1_INTC_s011
                                 + 0.927407 w_k1_KO_s011
                                 + 1.21803 w_k1_MAT_s011
                                 + 0.630041 w_k1_MSFT_s011
                                 + 0.902681 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.34022 w_k1_ADM_s011
                                 + 1.12415 w_k1_AXP_s011
                                 + 1.50351 w_k1_DIS_s011 + 1.2542 w_k1_FDX_s011
                                 + 1.39955 w_k1_IBM_s011
                                 + 1.02387 w_k1_INTC_s011
                                 + 1.58736 w_k1_KO_s011
                                 + 0.996653 w_k1_MAT_s011
                                 + 0.984761 w_k1_MSFT_s011
                                 + 1.28783 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.978849 w_k1_ADM_s011
                                 + 0.809307 w_k1_AXP_s011
                                 + 0.63771 w_k1_DIS_s011
                                 + 0.80745 w_k1_FDX_s011
                                 + 0.748333 w_k1_IBM_s011
                                 + 0.672182 w_k1_INTC_s011
                                 + 0.757777 w_k1_KO_s011
                                 + 0.692431 w_k1_MAT_s011
                                 + 0.68658 w_k1_MSFT_s011
                                 + 0.822799 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.04473 w_k1_ADM_s011
                                 + 0.971418 w_k1_AXP_s011
                                 + 1.03974 w_k1_DIS_s011
                                 + 1.05389 w_k1_FDX_s011
                                 + 1.02884 w_k1_IBM_s011
                                 + 0.919037 w_k1_INTC_s011
                                 + 1.00323 w_k1_KO_s011 + 1.13156 w_k1_MAT_s011
                                 + 1.01629 w_k1_MSFT_s011
                                 + 1.16568 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.18514 w_k1_ADM_s011
                                 + 0.85225 w_k1_AXP_s011
                                 + 0.810462 w_k1_DIS_s011
                                 + 0.84147 w_k1_FDX_s011
                                 + 1.00242 w_k1_IBM_s011
                                 + 0.883978 w_k1_INTC_s011
                                 + 0.910485 w_k1_KO_s011
                                 + 0.793012 w_k1_MAT_s011
                                 + 0.780679 w_k1_MSFT_s011
                                 + 0.973001 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.03543 w_k1_ADM_s012
                                 + 1.16732 w_k1_AXP_s012
                                 + 1.10046 w_k1_DIS_s012
                                 + 1.10299 w_k1_FDX_s012
                                 + 0.830703 w_k1_IBM_s012
                                 + 1.04007 w_k1_INTC_s012
                                 + 1.05176 w_k1_KO_s012 + 1.27296 w_k1_MAT_s012
                                 + 1.04097 w_k1_MSFT_s012
                                 + 1.06393 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.804858 w_k1_ADM_s012
                                 + 0.787122 w_k1_AXP_s012
                                 + 0.687966 w_k1_DIS_s012
                                 + 0.790067 w_k1_FDX_s012
                                 + 0.733489 w_k1_IBM_s012
                                 + 0.785999 w_k1_INTC_s012
                                 + 0.771775 w_k1_KO_s012
                                 + 0.942358 w_k1_MAT_s012
                                 + 0.885464 w_k1_MSFT_s012
                                 + 0.846172 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.129 w_k1_ADM_s012
                                 + 1.26884 w_k1_AXP_s012
                                 + 1.28537 w_k1_DIS_s012
                                 + 1.12817 w_k1_FDX_s012
                                 + 1.13429 w_k1_IBM_s012
                                 + 0.906981 w_k1_INTC_s012
                                 + 1.09327 w_k1_KO_s012 + 1.57559 w_k1_MAT_s012
                                 + 1.16713 w_k1_MSFT_s012
                                 + 1.21544 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.972572 w_k1_ADM_s012
                                 + 0.896261 w_k1_AXP_s012
                                 + 0.980756 w_k1_DIS_s012
                                 + 1.09201 w_k1_FDX_s012
                                 + 0.709828 w_k1_IBM_s012
                                 + 0.929627 w_k1_INTC_s012
                                 + 1.03878 w_k1_KO_s012
                                 + 0.900887 w_k1_MAT_s012
                                 + 1.08844 w_k1_MSFT_s012
                                 + 1.14391 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.39883 w_k1_ADM_s012
                                 + 1.41392 w_k1_AXP_s012
                                 + 1.98658 w_k1_DIS_s012
                                 + 1.36075 w_k1_FDX_s012
                                 + 1.19243 w_k1_IBM_s012
                                 + 1.29888 w_k1_INTC_s012
                                 + 1.71836 w_k1_KO_s012 + 1.73556 w_k1_MAT_s012
                                 + 1.65301 w_k1_MSFT_s012 + 1.3451 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.821619 w_k1_ADM_s012
                                 + 0.751385 w_k1_AXP_s012
                                 + 0.902015 w_k1_DIS_s012
                                 + 0.84872 w_k1_FDX_s012
                                 + 0.912037 w_k1_IBM_s012
                                 + 0.81053 w_k1_INTC_s012
                                 + 0.856124 w_k1_KO_s012
                                 + 0.799987 w_k1_MAT_s012
                                 + 0.896916 w_k1_MSFT_s012
                                 + 0.935495 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.06287 w_k1_ADM_s012
                                 + 0.921024 w_k1_AXP_s012
                                 + 0.973148 w_k1_DIS_s012
                                 + 0.875802 w_k1_FDX_s012
                                 + 1.24748 w_k1_IBM_s012
                                 + 0.893238 w_k1_INTC_s012
                                 + 1.17169 w_k1_KO_s012 + 1.1279 w_k1_MAT_s012
                                 + 0.936235 w_k1_MSFT_s012
                                 + 0.968898 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.974912 w_k1_ADM_s012
                                 + 1.10992 w_k1_AXP_s012
                                 + 0.660072 w_k1_DIS_s012
                                 + 0.781751 w_k1_FDX_s012
                                 + 1.03356 w_k1_IBM_s012
                                 + 0.705321 w_k1_INTC_s012
                                 + 0.810145 w_k1_KO_s012 + 1.6555 w_k1_MAT_s012
                                 + 0.747086 w_k1_MSFT_s012
                                 + 0.88265 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.21791 w_k1_ADM_s013
                                 + 0.758369 w_k1_AXP_s013
                                 + 0.984827 w_k1_DIS_s013
                                 + 1.07523 w_k1_FDX_s013
                                 + 0.730112 w_k1_IBM_s013
                                 + 1.11103 w_k1_INTC_s013
                                 + 0.870418 w_k1_KO_s013
                                 + 0.492823 w_k1_MAT_s013
                                 + 1.09867 w_k1_MSFT_s013
                                 + 0.901735 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.15463 w_k1_ADM_s013
                                 + 1.0721 w_k1_AXP_s013
                                 + 0.764984 w_k1_DIS_s013
                                 + 1.0172 w_k1_FDX_s013
                                 + 0.652269 w_k1_IBM_s013
                                 + 0.925819 w_k1_INTC_s013
                                 + 0.847997 w_k1_KO_s013
                                 + 1.08035 w_k1_MAT_s013
                                 + 1.01991 w_k1_MSFT_s013
                                 + 0.948981 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.19075 w_k1_ADM_s013
                                 + 1.55293 w_k1_AXP_s013
                                 + 1.78653 w_k1_DIS_s013
                                 + 1.41571 w_k1_FDX_s013
                                 + 1.45455 w_k1_IBM_s013
                                 + 1.19377 w_k1_INTC_s013
                                 + 1.53074 w_k1_KO_s013 + 1.74427 w_k1_MAT_s013
                                 + 1.44444 w_k1_MSFT_s013
                                 + 1.31266 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.61782 w_k1_ADM_s013
                                 + 1.09424 w_k1_AXP_s013 + 1.2584 w_k1_DIS_s013
                                 + 1.24556 w_k1_FDX_s013
                                 + 0.821525 w_k1_IBM_s013
                                 + 1.11679 w_k1_INTC_s013
                                 + 1.06135 w_k1_KO_s013
                                 + 0.588359 w_k1_MAT_s013
                                 + 1.21105 w_k1_MSFT_s013
                                 + 1.13888 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.20696 w_k1_ADM_s013
                                 + 1.28715 w_k1_AXP_s013
                                 + 1.08028 w_k1_DIS_s013
                                 + 1.17257 w_k1_FDX_s013
                                 + 1.27663 w_k1_IBM_s013
                                 + 1.11583 w_k1_INTC_s013
                                 + 1.03172 w_k1_KO_s013 + 1.6554 w_k1_MAT_s013
                                 + 1.11552 w_k1_MSFT_s013
                                 + 1.15058 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.767002 w_k1_ADM_s013
                                 + 0.761021 w_k1_AXP_s013
                                 + 0.5253 w_k1_DIS_s013
                                 + 0.729541 w_k1_FDX_s013
                                 + 0.538185 w_k1_IBM_s013
                                 + 0.830167 w_k1_INTC_s013
                                 + 0.555521 w_k1_KO_s013
                                 + 0.586356 w_k1_MAT_s013
                                 + 0.80387 w_k1_MSFT_s013
                                 + 0.700822 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.774684 w_k1_ADM_s013
                                 + 0.757722 w_k1_AXP_s013
                                 + 0.655513 w_k1_DIS_s013
                                 + 0.769892 w_k1_FDX_s013
                                 + 0.845474 w_k1_IBM_s013
                                 + 0.811807 w_k1_INTC_s013
                                 + 0.808068 w_k1_KO_s013
                                 + 0.829995 w_k1_MAT_s013
                                 + 0.794974 w_k1_MSFT_s013
                                 + 0.86449 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.0235 w_k1_ADM_s013
                                 + 0.875987 w_k1_AXP_s013
                                 + 0.961408 w_k1_DIS_s013
                                 + 0.935749 w_k1_FDX_s013
                                 + 1.17372 w_k1_IBM_s013
                                 + 1.02908 w_k1_INTC_s013
                                 + 0.972213 w_k1_KO_s013
                                 + 0.958254 w_k1_MAT_s013
                                 + 1.10038 w_k1_MSFT_s013
                                 + 0.953865 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.26897 w_k1_ADM_s014
                                 + 1.54269 w_k1_AXP_s014
                                 + 1.94037 w_k1_DIS_s014
                                 + 1.35865 w_k1_FDX_s014
                                 + 1.52724 w_k1_IBM_s014
                                 + 0.986125 w_k1_INTC_s014
                                 + 1.49972 w_k1_KO_s014 + 1.81363 w_k1_MAT_s014
                                 + 1.49418 w_k1_MSFT_s014
                                 + 1.38723 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.951874 w_k1_ADM_s014
                                 + 1.12788 w_k1_AXP_s014
                                 + 1.19571 w_k1_DIS_s014
                                 + 1.10923 w_k1_FDX_s014
                                 + 0.95754 w_k1_IBM_s014
                                 + 1.10518 w_k1_INTC_s014
                                 + 1.15052 w_k1_KO_s014
                                 + 0.812825 w_k1_MAT_s014
                                 + 1.14017 w_k1_MSFT_s014
                                 + 1.11859 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.893139 w_k1_ADM_s014
                                 + 0.826935 w_k1_AXP_s014
                                 + 0.825827 w_k1_DIS_s014
                                 + 0.89021 w_k1_FDX_s014
                                 + 0.773119 w_k1_IBM_s014
                                 + 0.817418 w_k1_INTC_s014
                                 + 0.750802 w_k1_KO_s014
                                 + 0.674837 w_k1_MAT_s014
                                 + 0.953286 w_k1_MSFT_s014
                                 + 0.877577 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.17339 w_k1_ADM_s014
                                 + 1.14624 w_k1_AXP_s014
                                 + 1.45385 w_k1_DIS_s014
                                 + 1.29455 w_k1_FDX_s014
                                 + 0.724124 w_k1_IBM_s014
                                 + 1.12439 w_k1_INTC_s014
                                 + 0.985778 w_k1_KO_s014
                                 + 0.751023 w_k1_MAT_s014
                                 + 1.27659 w_k1_MSFT_s014 + 1.1017 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.659971 w_k1_ADM_s014
                                 + 1.02178 w_k1_AXP_s014
                                 + 0.764449 w_k1_DIS_s014
                                 + 0.851329 w_k1_FDX_s014
                                 + 0.940574 w_k1_IBM_s014
                                 + 0.942162 w_k1_INTC_s014
                                 + 0.800782 w_k1_KO_s014
                                 + 0.933823 w_k1_MAT_s014
                                 + 0.853851 w_k1_MSFT_s014
                                 + 0.930474 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.641285 w_k1_ADM_s014
                                 + 0.79884 w_k1_AXP_s014
                                 + 0.694893 w_k1_DIS_s014
                                 + 0.810165 w_k1_FDX_s014
                                 + 0.579589 w_k1_IBM_s014
                                 + 0.751656 w_k1_INTC_s014
                                 + 0.74484 w_k1_KO_s014
                                 + 0.680474 w_k1_MAT_s014
                                 + 0.844079 w_k1_MSFT_s014
                                 + 0.817096 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.06873 w_k1_ADM_s014
                                 + 1.22986 w_k1_AXP_s014
                                 + 1.40951 w_k1_DIS_s014
                                 + 1.10741 w_k1_FDX_s014
                                 + 1.20735 w_k1_IBM_s014
                                 + 0.774877 w_k1_INTC_s014
                                 + 1.17454 w_k1_KO_s014 + 1.35476 w_k1_MAT_s014
                                 + 1.04021 w_k1_MSFT_s014
                                 + 1.05371 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.846175 w_k1_ADM_s014
                                 + 1.2353 w_k1_AXP_s014
                                 + 0.838239 w_k1_DIS_s014
                                 + 0.909732 w_k1_FDX_s014
                                 + 1.25462 w_k1_IBM_s014
                                 + 0.842406 w_k1_INTC_s014 + 0.922 w_k1_KO_s014
                                 + 1.39496 w_k1_MAT_s014
                                 + 0.903801 w_k1_MSFT_s014
                                 + 1.02553 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.93316 w_k1_ADM_s015
                                 + 0.741457 w_k1_AXP_s015
                                 + 0.814872 w_k1_DIS_s015
                                 + 0.828936 w_k1_FDX_s015
                                 + 0.886599 w_k1_IBM_s015
                                 + 0.814819 w_k1_INTC_s015
                                 + 0.796091 w_k1_KO_s015
                                 + 0.723007 w_k1_MAT_s015
                                 + 0.785958 w_k1_MSFT_s015
                                 + 0.964724 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.228 w_k1_ADM_s015
                                 + 1.17998 w_k1_AXP_s015
                                 + 1.30233 w_k1_DIS_s015
                                 + 1.19023 w_k1_FDX_s015
                                 + 1.01953 w_k1_IBM_s015
                                 + 1.24936 w_k1_INTC_s015
                                 + 1.13929 w_k1_KO_s015 + 1.20298 w_k1_MAT_s015
                                 + 1.09721 w_k1_MSFT_s015
                                 + 1.17741 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.17182 w_k1_ADM_s015
                                 + 1.33827 w_k1_AXP_s015
                                 + 1.77282 w_k1_DIS_s015
                                 + 1.37981 w_k1_FDX_s015
                                 + 1.40582 w_k1_IBM_s015
                                 + 1.3163 w_k1_INTC_s015 + 1.43027 w_k1_KO_s015
                                 + 1.51916 w_k1_MAT_s015
                                 + 1.19247 w_k1_MSFT_s015
                                 + 1.24431 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.887427 w_k1_ADM_s015
                                 + 0.713425 w_k1_AXP_s015
                                 + 0.975696 w_k1_DIS_s015
                                 + 0.96712 w_k1_FDX_s015
                                 + 0.630858 w_k1_IBM_s015
                                 + 0.962186 w_k1_INTC_s015
                                 + 0.916289 w_k1_KO_s015
                                 + 0.825482 w_k1_MAT_s015
                                 + 1.01584 w_k1_MSFT_s015
                                 + 0.964148 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.861177 w_k1_ADM_s015
                                 + 0.722038 w_k1_AXP_s015
                                 + 0.582289 w_k1_DIS_s015
                                 + 0.774886 w_k1_FDX_s015
                                 + 0.646588 w_k1_IBM_s015
                                 + 0.86962 w_k1_INTC_s015
                                 + 0.753802 w_k1_KO_s015
                                 + 0.694818 w_k1_MAT_s015
                                 + 0.655848 w_k1_MSFT_s015
                                 + 0.900592 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.995504 w_k1_ADM_s015
                                 + 1.04276 w_k1_AXP_s015
                                 + 1.06495 w_k1_DIS_s015
                                 + 1.10511 w_k1_FDX_s015
                                 + 0.809181 w_k1_IBM_s015
                                 + 1.18742 w_k1_INTC_s015
                                 + 0.930639 w_k1_KO_s015
                                 + 0.857409 w_k1_MAT_s015
                                 + 1.03603 w_k1_MSFT_s015 + 1.0837 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.848661 w_k1_ADM_s015
                                 + 1.0537 w_k1_AXP_s015
                                 + 0.983198 w_k1_DIS_s015
                                 + 0.951163 w_k1_FDX_s015
                                 + 1.2619 w_k1_IBM_s015
                                 + 0.917525 w_k1_INTC_s015
                                 + 0.995276 w_k1_KO_s015 + 1.2746 w_k1_MAT_s015
                                 + 0.869263 w_k1_MSFT_s015
                                 + 1.00391 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.10075 w_k1_ADM_s015
                                 + 1.0287 w_k1_AXP_s015
                                 + 0.708758 w_k1_DIS_s015
                                 + 0.902549 w_k1_FDX_s015
                                 + 0.9293 w_k1_IBM_s015
                                 + 0.836183 w_k1_INTC_s015
                                 + 0.766738 w_k1_KO_s015
                                 + 1.25976 w_k1_MAT_s015
                                 + 0.726638 w_k1_MSFT_s015
                                 + 0.996357 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.30227 w_k1_ADM_s016
                                 + 1.16244 w_k1_AXP_s016
                                 + 1.16906 w_k1_DIS_s016
                                 + 1.18357 w_k1_FDX_s016
                                 + 0.998058 w_k1_IBM_s016
                                 + 1.08094 w_k1_INTC_s016
                                 + 1.14942 w_k1_KO_s016 + 0.92076 w_k1_MAT_s016
                                 + 0.927216 w_k1_MSFT_s016
                                 + 1.16121 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.2213 w_k1_ADM_s016
                                 + 1.20965 w_k1_AXP_s016
                                 + 1.30383 w_k1_DIS_s016
                                 + 1.14047 w_k1_FDX_s016
                                 + 1.52947 w_k1_IBM_s016
                                 + 1.14736 w_k1_INTC_s016 + 1.2364 w_k1_KO_s016
                                 + 1.67123 w_k1_MAT_s016
                                 + 1.12373 w_k1_MSFT_s016
                                 + 1.00683 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.854733 w_k1_ADM_s016
                                 + 0.585589 w_k1_AXP_s016
                                 + 0.601805 w_k1_DIS_s016
                                 + 0.722383 w_k1_FDX_s016
                                 + 0.548328 w_k1_IBM_s016
                                 + 0.828059 w_k1_INTC_s016
                                 + 0.713434 w_k1_KO_s016
                                 + 0.336092 w_k1_MAT_s016
                                 + 0.785754 w_k1_MSFT_s016
                                 + 0.752064 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.09907 w_k1_ADM_s016
                                 + 0.727965 w_k1_AXP_s016
                                 + 0.884399 w_k1_DIS_s016
                                 + 0.917111 w_k1_FDX_s016
                                 + 0.690732 w_k1_IBM_s016
                                 + 0.868269 w_k1_INTC_s016
                                 + 0.868333 w_k1_KO_s016
                                 + 0.491906 w_k1_MAT_s016
                                 + 0.928931 w_k1_MSFT_s016
                                 + 0.889778 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.895204 w_k1_ADM_s016
                                 + 0.806782 w_k1_AXP_s016
                                 + 0.830115 w_k1_DIS_s016
                                 + 0.787897 w_k1_FDX_s016
                                 + 1.06189 w_k1_IBM_s016
                                 + 0.845619 w_k1_INTC_s016
                                 + 1.02207 w_k1_KO_s016
                                 + 0.765332 w_k1_MAT_s016
                                 + 0.82721 w_k1_MSFT_s016
                                 + 1.01608 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.05124 w_k1_ADM_s016
                                 + 0.995845 w_k1_AXP_s016
                                 + 1.39902 w_k1_DIS_s016
                                 + 1.14029 w_k1_FDX_s016
                                 + 0.724335 w_k1_IBM_s016
                                 + 1.04331 w_k1_INTC_s016
                                 + 0.969243 w_k1_KO_s016
                                 + 0.653093 w_k1_MAT_s016
                                 + 1.12455 w_k1_MSFT_s016
                                 + 1.09323 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.36169 w_k1_ADM_s016
                                 + 1.19347 w_k1_AXP_s016
                                 + 1.04521 w_k1_DIS_s016
                                 + 1.14683 w_k1_FDX_s016
                                 + 0.82852 w_k1_IBM_s016
                                 + 0.899876 w_k1_INTC_s016
                                 + 1.02628 w_k1_KO_s016
                                 + 0.749503 w_k1_MAT_s016
                                 + 0.902735 w_k1_MSFT_s016
                                 + 1.12225 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.42093 w_k1_ADM_s016
                                 + 1.11213 w_k1_AXP_s016
                                 + 1.45668 w_k1_DIS_s016
                                 + 1.23446 w_k1_FDX_s016
                                 + 1.16403 w_k1_IBM_s016
                                 + 1.01829 w_k1_INTC_s016
                                 + 1.42524 w_k1_KO_s016 + 1.2092 w_k1_MAT_s016
                                 + 1.12197 w_k1_MSFT_s016
                                 + 1.29902 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.23994 w_k1_ADM_s017
                                 + 1.31249 w_k1_AXP_s017
                                 + 1.59242 w_k1_DIS_s017
                                 + 1.34155 w_k1_FDX_s017
                                 + 1.08387 w_k1_IBM_s017
                                 + 1.3816 w_k1_INTC_s017 + 1.17859 w_k1_KO_s017
                                 + 1.27243 w_k1_MAT_s017
                                 + 1.39095 w_k1_MSFT_s017
                                 + 1.08827 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.920831 w_k1_ADM_s017
                                 + 1.07299 w_k1_AXP_s017
                                 + 1.00302 w_k1_DIS_s017
                                 + 0.884615 w_k1_FDX_s017
                                 + 1.57718 w_k1_IBM_s017
                                 + 1.12964 w_k1_INTC_s017
                                 + 1.23189 w_k1_KO_s017 + 1.41959 w_k1_MAT_s017
                                 + 1.01387 w_k1_MSFT_s017
                                 + 1.02633 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.83 w_k1_ADM_s017
                                 + 0.903378 w_k1_AXP_s017
                                 + 0.90797 w_k1_DIS_s017
                                 + 0.874483 w_k1_FDX_s017 + 1.252 w_k1_IBM_s017
                                 + 1.08974 w_k1_INTC_s017 + 1.0435 w_k1_KO_s017
                                 + 1.19746 w_k1_MAT_s017
                                 + 0.954928 w_k1_MSFT_s017
                                 + 1.01173 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.997222 w_k1_ADM_s017
                                 + 1.34387 w_k1_AXP_s017
                                 + 0.839363 w_k1_DIS_s017
                                 + 0.959607 w_k1_FDX_s017
                                 + 1.18464 w_k1_IBM_s017
                                 + 0.979226 w_k1_INTC_s017
                                 + 0.804973 w_k1_KO_s017
                                 + 1.27442 w_k1_MAT_s017
                                 + 0.793619 w_k1_MSFT_s017
                                 + 0.974703 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.20467 w_k1_ADM_s017
                                 + 1.07594 w_k1_AXP_s017
                                 + 1.14972 w_k1_DIS_s017
                                 + 1.05502 w_k1_FDX_s017
                                 + 1.02651 w_k1_IBM_s017
                                 + 1.33328 w_k1_INTC_s017 + 1.0267 w_k1_KO_s017
                                 + 0.776235 w_k1_MAT_s017
                                 + 1.00479 w_k1_MSFT_s017
                                 + 1.05838 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.16323 w_k1_ADM_s017
                                 + 1.571 w_k1_AXP_s017 + 1.4813 w_k1_DIS_s017
                                 + 1.21809 w_k1_FDX_s017
                                 + 1.73819 w_k1_IBM_s017
                                 + 1.28728 w_k1_INTC_s017
                                 + 1.41306 w_k1_KO_s017 + 1.79687 w_k1_MAT_s017
                                 + 1.13934 w_k1_MSFT_s017
                                 + 1.21339 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.893851 w_k1_ADM_s017
                                 + 0.830695 w_k1_AXP_s017
                                 + 0.847124 w_k1_DIS_s017
                                 + 0.889655 w_k1_FDX_s017
                                 + 0.75321 w_k1_IBM_s017
                                 + 1.04723 w_k1_INTC_s017
                                 + 0.810522 w_k1_KO_s017
                                 + 0.720562 w_k1_MAT_s017
                                 + 0.822127 w_k1_MSFT_s017
                                 + 0.839951 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.732224 w_k1_ADM_s017
                                 + 0.703141 w_k1_AXP_s017
                                 + 0.571518 w_k1_DIS_s017
                                 + 0.713222 w_k1_FDX_s017
                                 + 0.959865 w_k1_IBM_s017
                                 + 0.786922 w_k1_INTC_s017
                                 + 0.818974 w_k1_KO_s017
                                 + 0.86226 w_k1_MAT_s017
                                 + 0.729118 w_k1_MSFT_s017
                                 + 0.915695 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.20638 w_k1_ADM_s100
                                 + 1.74972 w_k1_AXP_s100
                                 + 1.53982 w_k1_DIS_s100
                                 + 1.44968 w_k1_FDX_s100
                                 + 1.12067 w_k1_IBM_s100
                                 + 1.2139 w_k1_INTC_s100 + 1.23107 w_k1_KO_s100
                                 + 1.40114 w_k1_MAT_s100
                                 + 1.13127 w_k1_MSFT_s100
                                 + 1.21279 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.19155 w_k1_ADM_s100
                                 + 1.18817 w_k1_AXP_s100
                                 + 0.838981 w_k1_DIS_s100
                                 + 0.965597 w_k1_FDX_s100
                                 + 0.848004 w_k1_IBM_s100
                                 + 0.887992 w_k1_INTC_s100
                                 + 0.85129 w_k1_KO_s100
                                 + 0.915946 w_k1_MAT_s100
                                 + 0.928239 w_k1_MSFT_s100
                                 + 1.02982 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1858 w_k1_ADM_s100
                                 + 1.314 w_k1_AXP_s100 + 1.00122 w_k1_DIS_s100
                                 + 0.987476 w_k1_FDX_s100
                                 + 1.28885 w_k1_IBM_s100
                                 + 0.951609 w_k1_INTC_s100
                                 + 1.10054 w_k1_KO_s100 + 2.14454 w_k1_MAT_s100
                                 + 0.895337 w_k1_MSFT_s100
                                 + 1.19139 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.977519 w_k1_ADM_s100
                                 + 1.06153 w_k1_AXP_s100
                                 + 0.899894 w_k1_DIS_s100
                                 + 0.864157 w_k1_FDX_s100
                                 + 1.4563 w_k1_IBM_s100
                                 + 0.885099 w_k1_INTC_s100
                                 + 0.996463 w_k1_KO_s100
                                 + 1.53261 w_k1_MAT_s100
                                 + 0.872338 w_k1_MSFT_s100
                                 + 0.983254 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.991406 w_k1_ADM_s100
                                 + 1.1218 w_k1_AXP_s100 + 1.36237 w_k1_DIS_s100
                                 + 1.16605 w_k1_FDX_s100
                                 + 1.04415 w_k1_IBM_s100
                                 + 1.34599 w_k1_INTC_s100
                                 + 1.28718 w_k1_KO_s100 + 1.07946 w_k1_MAT_s100
                                 + 1.2536 w_k1_MSFT_s100 + 1.11573 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.987873 w_k1_ADM_s100
                                 + 1.1115 w_k1_AXP_s100 + 1.0518 w_k1_DIS_s100
                                 + 1.15178 w_k1_FDX_s100
                                 + 0.755125 w_k1_IBM_s100
                                 + 1.20353 w_k1_INTC_s100
                                 + 0.868256 w_k1_KO_s100
                                 + 1.01058 w_k1_MAT_s100
                                 + 0.953549 w_k1_MSFT_s100
                                 + 0.948254 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.876067 w_k1_ADM_s100
                                 + 0.801759 w_k1_AXP_s100
                                 + 1.07494 w_k1_DIS_s100
                                 + 0.938911 w_k1_FDX_s100
                                 + 0.865464 w_k1_IBM_s100
                                 + 1.10613 w_k1_INTC_s100
                                 + 1.03111 w_k1_KO_s100
                                 + 0.933323 w_k1_MAT_s100
                                 + 0.990373 w_k1_MSFT_s100
                                 + 1.03284 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.780601 w_k1_ADM_s100
                                 + 0.802406 w_k1_AXP_s100
                                 + 0.742377 w_k1_DIS_s100
                                 + 0.806156 w_k1_FDX_s100
                                 + 0.7609 w_k1_IBM_s100
                                 + 1.09352 w_k1_INTC_s100
                                 + 0.822462 w_k1_KO_s100
                                 + 0.970258 w_k1_MAT_s100
                                 + 0.933379 w_k1_MSFT_s100
                                 + 0.905086 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.920507 w_k1_ADM_s101
                                 + 0.690843 w_k1_AXP_s101
                                 + 0.805245 w_k1_DIS_s101
                                 + 0.806849 w_k1_FDX_s101
                                 + 0.722228 w_k1_IBM_s101
                                 + 1.01743 w_k1_INTC_s101
                                 + 0.91179 w_k1_KO_s101
                                 + 0.609892 w_k1_MAT_s101
                                 + 1.06191 w_k1_MSFT_s101
                                 + 0.988413 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.44746 w_k1_ADM_s101
                                 + 0.992953 w_k1_AXP_s101
                                 + 1.5734 w_k1_DIS_s101 + 1.28423 w_k1_FDX_s101
                                 + 1.08512 w_k1_IBM_s101
                                 + 1.17932 w_k1_INTC_s101
                                 + 1.30753 w_k1_KO_s101
                                 + 0.728922 w_k1_MAT_s101
                                 + 1.42686 w_k1_MSFT_s101
                                 + 1.11319 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.22135 w_k1_ADM_s101
                                 + 1.21358 w_k1_AXP_s101
                                 + 1.28092 w_k1_DIS_s101
                                 + 1.07519 w_k1_FDX_s101
                                 + 1.17921 w_k1_IBM_s101
                                 + 0.934662 w_k1_INTC_s101
                                 + 1.31126 w_k1_KO_s101 + 1.248 w_k1_MAT_s101
                                 + 1.2197 w_k1_MSFT_s101 + 1.02069 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.14583 w_k1_ADM_s101
                                 + 1.48086 w_k1_AXP_s101
                                 + 1.29865 w_k1_DIS_s101
                                 + 1.04859 w_k1_FDX_s101
                                 + 1.52615 w_k1_IBM_s101
                                 + 1.13382 w_k1_INTC_s101
                                 + 1.33748 w_k1_KO_s101 + 1.90382 w_k1_MAT_s101
                                 + 1.16162 w_k1_MSFT_s101
                                 + 1.22255 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.90506 w_k1_ADM_s101
                                 + 0.82014 w_k1_AXP_s101
                                 + 0.889722 w_k1_DIS_s101
                                 + 0.885005 w_k1_FDX_s101
                                 + 1.19567 w_k1_IBM_s101
                                 + 0.89472 w_k1_INTC_s101
                                 + 1.11554 w_k1_KO_s101 + 1.17612 w_k1_MAT_s101
                                 + 0.999463 w_k1_MSFT_s101
                                 + 0.941475 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.04186 w_k1_ADM_s101
                                 + 0.883647 w_k1_AXP_s101
                                 + 0.679662 w_k1_DIS_s101
                                 + 0.879767 w_k1_FDX_s101
                                 + 0.79525 w_k1_IBM_s101
                                 + 0.861922 w_k1_INTC_s101
                                 + 0.794725 w_k1_KO_s101
                                 + 0.951294 w_k1_MAT_s101
                                 + 0.899806 w_k1_MSFT_s101
                                 + 0.866687 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.887556 w_k1_ADM_s101
                                 + 0.825127 w_k1_AXP_s101
                                 + 0.600519 w_k1_DIS_s101
                                 + 0.757757 w_k1_FDX_s101
                                 + 0.891927 w_k1_IBM_s101
                                 + 0.814461 w_k1_INTC_s101
                                 + 0.851729 w_k1_KO_s101
                                 + 1.01877 w_k1_MAT_s101
                                 + 0.873195 w_k1_MSFT_s101
                                 + 0.874873 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.20457 w_k1_ADM_s101
                                 + 0.999032 w_k1_AXP_s101
                                 + 0.985968 w_k1_DIS_s101
                                 + 1.03572 w_k1_FDX_s101
                                 + 0.930605 w_k1_IBM_s101
                                 + 0.974813 w_k1_INTC_s101
                                 + 0.979604 w_k1_KO_s101 + 1.2907 w_k1_MAT_s101
                                 + 1.14657 w_k1_MSFT_s101
                                 + 1.02271 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.944245 w_k1_ADM_s102
                                 + 0.763858 w_k1_AXP_s102
                                 + 0.924081 w_k1_DIS_s102
                                 + 0.899208 w_k1_FDX_s102
                                 + 0.706257 w_k1_IBM_s102
                                 + 1.14498 w_k1_INTC_s102
                                 + 0.885068 w_k1_KO_s102
                                 + 0.455484 w_k1_MAT_s102
                                 + 0.989573 w_k1_MSFT_s102
                                 + 0.870955 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.14063 w_k1_ADM_s102
                                 + 1.15045 w_k1_AXP_s102
                                 + 1.23531 w_k1_DIS_s102
                                 + 1.21699 w_k1_FDX_s102
                                 + 0.795513 w_k1_IBM_s102
                                 + 1.29195 w_k1_INTC_s102
                                 + 0.974565 w_k1_KO_s102
                                 + 0.953763 w_k1_MAT_s102
                                 + 1.29297 w_k1_MSFT_s102
                                 + 1.02993 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.35216 w_k1_ADM_s102
                                 + 1.5825 w_k1_AXP_s102 + 1.6471 w_k1_DIS_s102
                                 + 1.43136 w_k1_FDX_s102
                                 + 1.34617 w_k1_IBM_s102
                                 + 1.54506 w_k1_INTC_s102 + 1.4555 w_k1_KO_s102
                                 + 1.65227 w_k1_MAT_s102
                                 + 1.61791 w_k1_MSFT_s102
                                 + 1.19176 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05618 w_k1_ADM_s102
                                 + 1.15668 w_k1_AXP_s102
                                 + 0.971678 w_k1_DIS_s102
                                 + 0.93075 w_k1_FDX_s102
                                 + 1.00077 w_k1_IBM_s102
                                 + 1.13239 w_k1_INTC_s102
                                 + 0.961273 w_k1_KO_s102
                                 + 1.33458 w_k1_MAT_s102
                                 + 0.97847 w_k1_MSFT_s102
                                 + 0.949538 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.2153 w_k1_ADM_s102
                                 + 1.19706 w_k1_AXP_s102
                                 + 1.14867 w_k1_DIS_s102
                                 + 1.05253 w_k1_FDX_s102
                                 + 1.39269 w_k1_IBM_s102
                                 + 1.24179 w_k1_INTC_s102 + 1.3181 w_k1_KO_s102
                                 + 1.33364 w_k1_MAT_s102
                                 + 1.12371 w_k1_MSFT_s102 + 1.1316 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.999732 w_k1_ADM_s102
                                 + 0.597118 w_k1_AXP_s102
                                 + 0.686872 w_k1_DIS_s102
                                 + 0.684483 w_k1_FDX_s102
                                 + 0.820703 w_k1_IBM_s102
                                 + 0.835858 w_k1_INTC_s102
                                 + 0.724785 w_k1_KO_s102
                                 + 0.59669 w_k1_MAT_s102
                                 + 0.835601 w_k1_MSFT_s102
                                 + 0.850264 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.20875 w_k1_ADM_s102
                                 + 0.901285 w_k1_AXP_s102
                                 + 0.830669 w_k1_DIS_s102
                                 + 0.902057 w_k1_FDX_s102
                                 + 0.845912 w_k1_IBM_s102
                                 + 0.829059 w_k1_INTC_s102
                                 + 0.837805 w_k1_KO_s102
                                 + 0.931403 w_k1_MAT_s102
                                 + 0.942735 w_k1_MSFT_s102
                                 + 0.991132 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.10566 w_k1_ADM_s102
                                 + 0.849053 w_k1_AXP_s102
                                 + 0.820687 w_k1_DIS_s102
                                 + 0.730978 w_k1_FDX_s102
                                 + 1.27428 w_k1_IBM_s102
                                 + 0.901362 w_k1_INTC_s102
                                 + 1.03495 w_k1_KO_s102 + 1.29655 w_k1_MAT_s102
                                 + 0.924232 w_k1_MSFT_s102
                                 + 0.935056 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.24981 w_k1_ADM_s103
                                 + 0.86134 w_k1_AXP_s103 + 1.1729 w_k1_DIS_s103
                                 + 1.05735 w_k1_FDX_s103
                                 + 0.954444 w_k1_IBM_s103
                                 + 0.998042 w_k1_INTC_s103
                                 + 0.854284 w_k1_KO_s103
                                 + 0.762874 w_k1_MAT_s103
                                 + 1.11545 w_k1_MSFT_s103
                                 + 1.04822 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.983028 w_k1_ADM_s103
                                 + 1.28583 w_k1_AXP_s103
                                 + 1.23122 w_k1_DIS_s103
                                 + 1.23852 w_k1_FDX_s103
                                 + 1.15789 w_k1_IBM_s103
                                 + 1.1425 w_k1_INTC_s103
                                 + 0.998612 w_k1_KO_s103
                                 + 1.30063 w_k1_MAT_s103
                                 + 1.13404 w_k1_MSFT_s103
                                 + 1.10081 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.28113 w_k1_ADM_s103
                                 + 1.30516 w_k1_AXP_s103
                                 + 1.08722 w_k1_DIS_s103
                                 + 1.12912 w_k1_FDX_s103
                                 + 0.76954 w_k1_IBM_s103
                                 + 1.22298 w_k1_INTC_s103
                                 + 0.952739 w_k1_KO_s103
                                 + 0.882005 w_k1_MAT_s103
                                 + 1.13913 w_k1_MSFT_s103 + 1.1049 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.46426 w_k1_ADM_s103
                                 + 1.48236 w_k1_AXP_s103
                                 + 1.54853 w_k1_DIS_s103
                                 + 1.54974 w_k1_FDX_s103
                                 + 1.17217 w_k1_IBM_s103
                                 + 1.48218 w_k1_INTC_s103
                                 + 1.11855 w_k1_KO_s103 + 1.1121 w_k1_MAT_s103
                                 + 1.19115 w_k1_MSFT_s103
                                 + 1.15686 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.00498 w_k1_ADM_s103
                                 + 0.810311 w_k1_AXP_s103
                                 + 0.688021 w_k1_DIS_s103
                                 + 0.88758 w_k1_FDX_s103
                                 + 0.731735 w_k1_IBM_s103
                                 + 0.798448 w_k1_INTC_s103
                                 + 0.663929 w_k1_KO_s103
                                 + 0.785835 w_k1_MAT_s103
                                 + 0.862132 w_k1_MSFT_s103
                                 + 0.875699 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.684708 w_k1_ADM_s103
                                 + 0.51486 w_k1_AXP_s103
                                 + 0.504949 w_k1_DIS_s103
                                 + 0.63969 w_k1_FDX_s103
                                 + 0.855729 w_k1_IBM_s103
                                 + 0.813815 w_k1_INTC_s103
                                 + 0.566249 w_k1_KO_s103
                                 + 0.602988 w_k1_MAT_s103
                                 + 0.685805 w_k1_MSFT_s103
                                 + 0.753982 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.13364 w_k1_ADM_s103
                                 + 1.07889 w_k1_AXP_s103
                                 + 1.15918 w_k1_DIS_s103 + 1.0066 w_k1_FDX_s103
                                 + 1.59191 w_k1_IBM_s103
                                 + 0.993304 w_k1_INTC_s103
                                 + 1.0299 w_k1_KO_s103 + 1.30613 w_k1_MAT_s103
                                 + 0.890167 w_k1_MSFT_s103
                                 + 1.08998 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01783 w_k1_ADM_s103
                                 + 0.900971 w_k1_AXP_s103
                                 + 0.916001 w_k1_DIS_s103
                                 + 0.928979 w_k1_FDX_s103
                                 + 1.1019 w_k1_IBM_s103
                                 + 0.959609 w_k1_INTC_s103
                                 + 0.929509 w_k1_KO_s103
                                 + 0.929007 w_k1_MAT_s103
                                 + 0.971527 w_k1_MSFT_s103
                                 + 1.0755 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.62782 w_k1_ADM_s104
                                 + 1.11317 w_k1_AXP_s104
                                 + 1.49529 w_k1_DIS_s104
                                 + 1.41935 w_k1_FDX_s104
                                 + 1.06736 w_k1_IBM_s104
                                 + 1.19034 w_k1_INTC_s104
                                 + 1.33334 w_k1_KO_s104 + 1.08139 w_k1_MAT_s104
                                 + 1.33574 w_k1_MSFT_s104
                                 + 1.20744 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.935782 w_k1_ADM_s104
                                 + 0.977847 w_k1_AXP_s104
                                 + 0.775892 w_k1_DIS_s104
                                 + 0.772088 w_k1_FDX_s104
                                 + 1.07679 w_k1_IBM_s104
                                 + 0.817761 w_k1_INTC_s104
                                 + 0.944365 w_k1_KO_s104
                                 + 1.16859 w_k1_MAT_s104
                                 + 0.711628 w_k1_MSFT_s104
                                 + 0.962987 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.38395 w_k1_ADM_s104
                                 + 1.19713 w_k1_AXP_s104
                                 + 1.19919 w_k1_DIS_s104
                                 + 1.10007 w_k1_FDX_s104
                                 + 1.30887 w_k1_IBM_s104 + 1.111 w_k1_INTC_s104
                                 + 1.20216 w_k1_KO_s104 + 1.39715 w_k1_MAT_s104
                                 + 0.942498 w_k1_MSFT_s104
                                 + 1.13241 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.912448 w_k1_ADM_s104
                                 + 0.592451 w_k1_AXP_s104
                                 + 0.573573 w_k1_DIS_s104
                                 + 0.730211 w_k1_FDX_s104
                                 + 0.785681 w_k1_IBM_s104
                                 + 0.712935 w_k1_INTC_s104
                                 + 0.695084 w_k1_KO_s104
                                 + 0.531168 w_k1_MAT_s104
                                 + 0.717778 w_k1_MSFT_s104
                                 + 0.899601 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.24636 w_k1_ADM_s104
                                 + 0.994043 w_k1_AXP_s104
                                 + 0.808713 w_k1_DIS_s104
                                 + 0.924484 w_k1_FDX_s104
                                 + 1.31769 w_k1_IBM_s104
                                 + 0.894606 w_k1_INTC_s104
                                 + 1.00141 w_k1_KO_s104 + 1.15433 w_k1_MAT_s104
                                 + 0.832361 w_k1_MSFT_s104
                                 + 0.948936 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.42326 w_k1_ADM_s104
                                 + 1.0522 w_k1_AXP_s104 + 1.0526 w_k1_DIS_s104
                                 + 1.11283 w_k1_FDX_s104
                                 + 0.797232 w_k1_IBM_s104
                                 + 0.986376 w_k1_INTC_s104
                                 + 0.976499 w_k1_KO_s104
                                 + 0.662768 w_k1_MAT_s104
                                 + 1.05725 w_k1_MSFT_s104
                                 + 1.05254 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.60408 w_k1_ADM_s104
                                 + 1.82399 w_k1_AXP_s104
                                 + 1.40703 w_k1_DIS_s104
                                 + 1.40552 w_k1_FDX_s104
                                 + 1.30618 w_k1_IBM_s104
                                 + 1.15456 w_k1_INTC_s104 + 1.236 w_k1_KO_s104
                                 + 1.57421 w_k1_MAT_s104
                                 + 1.28506 w_k1_MSFT_s104
                                 + 1.35423 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.17151 w_k1_ADM_s104
                                 + 0.94882 w_k1_AXP_s104
                                 + 0.735257 w_k1_DIS_s104
                                 + 0.941323 w_k1_FDX_s104
                                 + 0.820293 w_k1_IBM_s104
                                 + 0.907399 w_k1_INTC_s104
                                 + 0.856132 w_k1_KO_s104
                                 + 0.795599 w_k1_MAT_s104
                                 + 0.859329 w_k1_MSFT_s104
                                 + 1.00262 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.00968 w_k1_ADM_s105
                                 + 1.02944 w_k1_AXP_s105
                                 + 0.903566 w_k1_DIS_s105
                                 + 1.00342 w_k1_FDX_s105
                                 + 0.882986 w_k1_IBM_s105
                                 + 0.82018 w_k1_INTC_s105
                                 + 0.816678 w_k1_KO_s105
                                 + 0.860452 w_k1_MAT_s105
                                 + 1.08202 w_k1_MSFT_s105
                                 + 0.858544 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06067 w_k1_ADM_s105
                                 + 1.13188 w_k1_AXP_s105
                                 + 1.16718 w_k1_DIS_s105
                                 + 1.03009 w_k1_FDX_s105
                                 + 1.14683 w_k1_IBM_s105
                                 + 0.980942 w_k1_INTC_s105
                                 + 1.01865 w_k1_KO_s105 + 1.076 w_k1_MAT_s105
                                 + 1.08666 w_k1_MSFT_s105
                                 + 1.02587 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.15083 w_k1_ADM_s105
                                 + 1.15318 w_k1_AXP_s105
                                 + 1.35438 w_k1_DIS_s105
                                 + 1.13973 w_k1_FDX_s105
                                 + 1.52136 w_k1_IBM_s105
                                 + 1.12092 w_k1_INTC_s105
                                 + 1.28482 w_k1_KO_s105 + 1.44336 w_k1_MAT_s105
                                 + 1.23161 w_k1_MSFT_s105
                                 + 1.13028 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.24571 w_k1_ADM_s105
                                 + 1.24131 w_k1_AXP_s105
                                 + 1.38835 w_k1_DIS_s105
                                 + 1.34262 w_k1_FDX_s105
                                 + 0.824982 w_k1_IBM_s105
                                 + 1.08289 w_k1_INTC_s105
                                 + 1.12165 w_k1_KO_s105 + 0.85714 w_k1_MAT_s105
                                 + 1.30005 w_k1_MSFT_s105
                                 + 1.08133 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.89393 w_k1_ADM_s105
                                 + 0.681269 w_k1_AXP_s105
                                 + 0.639863 w_k1_DIS_s105
                                 + 0.719987 w_k1_FDX_s105
                                 + 0.826655 w_k1_IBM_s105
                                 + 0.609697 w_k1_INTC_s105
                                 + 0.751284 w_k1_KO_s105
                                 + 0.93047 w_k1_MAT_s105
                                 + 0.926441 w_k1_MSFT_s105
                                 + 0.813764 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.735133 w_k1_ADM_s105
                                 + 0.71219 w_k1_AXP_s105
                                 + 0.91705 w_k1_DIS_s105
                                 + 0.799424 w_k1_FDX_s105
                                 + 1.04367 w_k1_IBM_s105
                                 + 0.82458 w_k1_INTC_s105
                                 + 1.11234 w_k1_KO_s105 + 1.04532 w_k1_MAT_s105
                                 + 1.02899 w_k1_MSFT_s105
                                 + 0.900987 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.871193 w_k1_ADM_s105
                                 + 0.83231 w_k1_AXP_s105
                                 + 0.796703 w_k1_DIS_s105
                                 + 0.847378 w_k1_FDX_s105
                                 + 0.634037 w_k1_IBM_s105
                                 + 0.740568 w_k1_INTC_s105
                                 + 0.689924 w_k1_KO_s105
                                 + 0.883309 w_k1_MAT_s105
                                 + 0.919556 w_k1_MSFT_s105
                                 + 0.826562 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.820282 w_k1_ADM_s105
                                 + 0.712481 w_k1_AXP_s105
                                 + 1.10225 w_k1_DIS_s105
                                 + 0.991033 w_k1_FDX_s105
                                 + 0.664634 w_k1_IBM_s105
                                 + 0.68189 w_k1_INTC_s105
                                 + 0.885247 w_k1_KO_s105
                                 + 0.617782 w_k1_MAT_s105
                                 + 1.04341 w_k1_MSFT_s105
                                 + 0.882362 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.54462 w_k1_ADM_s106
                                 + 1.5901 w_k1_AXP_s106 + 1.70213 w_k1_DIS_s106
                                 + 1.49595 w_k1_FDX_s106 + 1.1424 w_k1_IBM_s106
                                 + 1.1815 w_k1_INTC_s106 + 1.16423 w_k1_KO_s106
                                 + 0.869467 w_k1_MAT_s106
                                 + 1.2308 w_k1_MSFT_s106 + 1.2259 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.11246 w_k1_ADM_s106
                                 + 0.787577 w_k1_AXP_s106
                                 + 1.26177 w_k1_DIS_s106
                                 + 1.11977 w_k1_FDX_s106
                                 + 1.09345 w_k1_IBM_s106
                                 + 1.07071 w_k1_INTC_s106
                                 + 1.21504 w_k1_KO_s106
                                 + 0.568017 w_k1_MAT_s106
                                 + 1.14602 w_k1_MSFT_s106
                                 + 1.07868 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.827941 w_k1_ADM_s106
                                 + 0.872437 w_k1_AXP_s106
                                 + 0.629218 w_k1_DIS_s106
                                 + 0.759066 w_k1_FDX_s106
                                 + 1.4018 w_k1_IBM_s106
                                 + 0.786919 w_k1_INTC_s106
                                 + 0.846364 w_k1_KO_s106
                                 + 1.41645 w_k1_MAT_s106
                                 + 0.786549 w_k1_MSFT_s106
                                 + 0.893961 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12391 w_k1_ADM_s106
                                 + 1.181 w_k1_AXP_s106 + 1.02329 w_k1_DIS_s106
                                 + 1.16702 w_k1_FDX_s106
                                 + 0.950361 w_k1_IBM_s106
                                 + 1.0509 w_k1_INTC_s106 + 1.01256 w_k1_KO_s106
                                 + 0.97192 w_k1_MAT_s106
                                 + 1.00662 w_k1_MSFT_s106 + 1.0096 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.22578 w_k1_ADM_s106
                                 + 1.27751 w_k1_AXP_s106
                                 + 1.16535 w_k1_DIS_s106
                                 + 1.01887 w_k1_FDX_s106
                                 + 1.62923 w_k1_IBM_s106
                                 + 1.21325 w_k1_INTC_s106
                                 + 1.37435 w_k1_KO_s106 + 1.60984 w_k1_MAT_s106
                                 + 1.0521 w_k1_MSFT_s106 + 1.13255 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.21611 w_k1_ADM_s106
                                 + 1.22855 w_k1_AXP_s106
                                 + 1.10664 w_k1_DIS_s106
                                 + 1.04055 w_k1_FDX_s106
                                 + 1.27947 w_k1_IBM_s106
                                 + 1.03893 w_k1_INTC_s106
                                 + 1.08975 w_k1_KO_s106 + 1.30361 w_k1_MAT_s106
                                 + 0.959262 w_k1_MSFT_s106
                                 + 1.00848 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.35646 w_k1_ADM_s106
                                 + 1.18329 w_k1_AXP_s106
                                 + 1.16376 w_k1_DIS_s106
                                 + 1.30624 w_k1_FDX_s106
                                 + 0.706241 w_k1_IBM_s106
                                 + 1.16717 w_k1_INTC_s106
                                 + 0.959574 w_k1_KO_s106
                                 + 0.556519 w_k1_MAT_s106
                                 + 1.12996 w_k1_MSFT_s106
                                 + 0.985694 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.861425 w_k1_ADM_s106
                                 + 0.708039 w_k1_AXP_s106
                                 + 0.731012 w_k1_DIS_s106
                                 + 0.787491 w_k1_FDX_s106
                                 + 0.760041 w_k1_IBM_s106
                                 + 0.878592 w_k1_INTC_s106
                                 + 0.86936 w_k1_KO_s106
                                 + 0.678393 w_k1_MAT_s106
                                 + 0.822771 w_k1_MSFT_s106
                                 + 0.799678 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.927325 w_k1_ADM_s107
                                 + 1.07175 w_k1_AXP_s107
                                 + 1.09158 w_k1_DIS_s107
                                 + 1.22873 w_k1_FDX_s107
                                 + 0.761608 w_k1_IBM_s107
                                 + 1.16643 w_k1_INTC_s107
                                 + 0.809877 w_k1_KO_s107
                                 + 0.744242 w_k1_MAT_s107
                                 + 1.05957 w_k1_MSFT_s107
                                 + 1.03887 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.852042 w_k1_ADM_s107
                                 + 0.841171 w_k1_AXP_s107
                                 + 1.12409 w_k1_DIS_s107
                                 + 0.951474 w_k1_FDX_s107
                                 + 1.40309 w_k1_IBM_s107
                                 + 1.01658 w_k1_INTC_s107
                                 + 1.06896 w_k1_KO_s107 + 1.27344 w_k1_MAT_s107
                                 + 0.990875 w_k1_MSFT_s107
                                 + 0.990787 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.886492 w_k1_ADM_s107
                                 + 1.0033 w_k1_AXP_s107 + 1.41737 w_k1_DIS_s107
                                 + 1.22951 w_k1_FDX_s107
                                 + 1.03517 w_k1_IBM_s107
                                 + 1.14931 w_k1_INTC_s107
                                 + 1.09539 w_k1_KO_s107 + 1.48114 w_k1_MAT_s107
                                 + 1.31801 w_k1_MSFT_s107
                                 + 1.16051 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.21458 w_k1_ADM_s107
                                 + 1.36248 w_k1_AXP_s107
                                 + 1.55434 w_k1_DIS_s107
                                 + 1.54186 w_k1_FDX_s107
                                 + 0.873972 w_k1_IBM_s107
                                 + 1.22605 w_k1_INTC_s107
                                 + 1.09545 w_k1_KO_s107
                                 + 0.979853 w_k1_MAT_s107
                                 + 1.44622 w_k1_MSFT_s107
                                 + 1.15142 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.972667 w_k1_ADM_s107
                                 + 0.900408 w_k1_AXP_s107
                                 + 0.850806 w_k1_DIS_s107
                                 + 0.950484 w_k1_FDX_s107
                                 + 0.995721 w_k1_IBM_s107
                                 + 0.792955 w_k1_INTC_s107
                                 + 0.889129 w_k1_KO_s107
                                 + 0.909615 w_k1_MAT_s107
                                 + 0.945748 w_k1_MSFT_s107
                                 + 0.916919 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.769742 w_k1_ADM_s107
                                 + 0.72971 w_k1_AXP_s107
                                 + 0.755462 w_k1_DIS_s107
                                 + 0.840273 w_k1_FDX_s107
                                 + 0.789777 w_k1_IBM_s107
                                 + 0.912959 w_k1_INTC_s107
                                 + 0.711698 w_k1_KO_s107
                                 + 0.87487 w_k1_MAT_s107
                                 + 0.8699 w_k1_MSFT_s107
                                 + 0.906316 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.14471 w_k1_ADM_s107
                                 + 1.11679 w_k1_AXP_s107
                                 + 1.06391 w_k1_DIS_s107
                                 + 1.13961 w_k1_FDX_s107
                                 + 1.21523 w_k1_IBM_s107
                                 + 1.00712 w_k1_INTC_s107
                                 + 1.01977 w_k1_KO_s107 + 1.14705 w_k1_MAT_s107
                                 + 1.00986 w_k1_MSFT_s107
                                 + 1.02157 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.964561 w_k1_ADM_s107
                                 + 1.3465 w_k1_AXP_s107 + 1.31054 w_k1_DIS_s107
                                 + 1.11676 w_k1_FDX_s107
                                 + 1.47491 w_k1_IBM_s107
                                 + 1.18673 w_k1_INTC_s107
                                 + 1.40111 w_k1_KO_s107 + 1.85871 w_k1_MAT_s107
                                 + 1.09036 w_k1_MSFT_s107
                                 + 1.19532 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.744666 w_k1_ADM_s110
                                 + 0.906603 w_k1_AXP_s110
                                 + 0.928971 w_k1_DIS_s110
                                 + 0.934076 w_k1_FDX_s110
                                 + 0.930845 w_k1_IBM_s110
                                 + 1.02719 w_k1_INTC_s110
                                 + 0.971972 w_k1_KO_s110
                                 + 0.856827 w_k1_MAT_s110
                                 + 0.874767 w_k1_MSFT_s110
                                 + 0.973953 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.826605 w_k1_ADM_s110
                                 + 0.772343 w_k1_AXP_s110
                                 + 0.902568 w_k1_DIS_s110
                                 + 0.842372 w_k1_FDX_s110
                                 + 0.879274 w_k1_IBM_s110
                                 + 0.912555 w_k1_INTC_s110
                                 + 1.07492 w_k1_KO_s110
                                 + 0.720345 w_k1_MAT_s110
                                 + 0.842796 w_k1_MSFT_s110
                                 + 0.85545 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.40942 w_k1_ADM_s110
                                 + 1.46712 w_k1_AXP_s110
                                 + 1.42438 w_k1_DIS_s110
                                 + 1.30007 w_k1_FDX_s110 + 1.1287 w_k1_IBM_s110
                                 + 1.33318 w_k1_INTC_s110
                                 + 1.34624 w_k1_KO_s110 + 1.38316 w_k1_MAT_s110
                                 + 1.20553 w_k1_MSFT_s110
                                 + 1.08843 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.846907 w_k1_ADM_s110
                                 + 1.2239 w_k1_AXP_s110 + 1.21233 w_k1_DIS_s110
                                 + 1.0106 w_k1_FDX_s110 + 1.58513 w_k1_IBM_s110
                                 + 1.07134 w_k1_INTC_s110
                                 + 1.34098 w_k1_KO_s110 + 1.92181 w_k1_MAT_s110
                                 + 0.932286 w_k1_MSFT_s110
                                 + 1.08476 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.927026 w_k1_ADM_s110
                                 + 0.869062 w_k1_AXP_s110
                                 + 0.833583 w_k1_DIS_s110
                                 + 0.922057 w_k1_FDX_s110
                                 + 0.675835 w_k1_IBM_s110
                                 + 1.10369 w_k1_INTC_s110
                                 + 0.865421 w_k1_KO_s110
                                 + 0.484819 w_k1_MAT_s110
                                 + 0.794448 w_k1_MSFT_s110
                                 + 0.928274 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.996248 w_k1_ADM_s110
                                 + 1.00923 w_k1_AXP_s110
                                 + 1.39693 w_k1_DIS_s110
                                 + 1.16707 w_k1_FDX_s110
                                 + 1.24154 w_k1_IBM_s110
                                 + 1.14258 w_k1_INTC_s110
                                 + 1.56224 w_k1_KO_s110 + 1.16376 w_k1_MAT_s110
                                 + 1.20201 w_k1_MSFT_s110
                                 + 1.18996 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.933456 w_k1_ADM_s110
                                 + 1.06253 w_k1_AXP_s110
                                 + 0.848479 w_k1_DIS_s110
                                 + 0.906425 w_k1_FDX_s110
                                 + 1.1556 w_k1_IBM_s110
                                 + 0.897011 w_k1_INTC_s110
                                 + 1.13067 w_k1_KO_s110 + 1.33227 w_k1_MAT_s110
                                 + 0.797258 w_k1_MSFT_s110
                                 + 1.04512 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.17159 w_k1_ADM_s110
                                 + 1.00687 w_k1_AXP_s110
                                 + 1.31365 w_k1_DIS_s110
                                 + 1.25808 w_k1_FDX_s110
                                 + 0.808099 w_k1_IBM_s110
                                 + 1.33009 w_k1_INTC_s110 + 1.1678 w_k1_KO_s110
                                 + 0.911519 w_k1_MAT_s110
                                 + 1.31464 w_k1_MSFT_s110
                                 + 1.15617 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.08007 w_k1_ADM_s111
                                 + 1.63937 w_k1_AXP_s111
                                 + 1.38264 w_k1_DIS_s111
                                 + 1.27715 w_k1_FDX_s111
                                 + 1.67063 w_k1_IBM_s111
                                 + 1.17708 w_k1_INTC_s111
                                 + 1.45306 w_k1_KO_s111 + 1.93888 w_k1_MAT_s111
                                 + 1.07123 w_k1_MSFT_s111
                                 + 1.54644 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.07765 w_k1_ADM_s111
                                 + 1.22704 w_k1_AXP_s111
                                 + 0.907568 w_k1_DIS_s111
                                 + 1.02248 w_k1_FDX_s111
                                 + 1.04519 w_k1_IBM_s111
                                 + 0.88125 w_k1_INTC_s111
                                 + 0.921994 w_k1_KO_s111
                                 + 1.13726 w_k1_MAT_s111
                                 + 0.96729 w_k1_MSFT_s111
                                 + 1.07284 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.755188 w_k1_ADM_s111
                                 + 0.860554 w_k1_AXP_s111
                                 + 0.721279 w_k1_DIS_s111
                                 + 0.965295 w_k1_FDX_s111
                                 + 0.754784 w_k1_IBM_s111
                                 + 0.742138 w_k1_INTC_s111
                                 + 0.740416 w_k1_KO_s111
                                 + 0.666546 w_k1_MAT_s111
                                 + 0.804245 w_k1_MSFT_s111
                                 + 0.856923 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.767675 w_k1_ADM_s111
                                 + 0.888618 w_k1_AXP_s111
                                 + 0.881049 w_k1_DIS_s111
                                 + 0.900481 w_k1_FDX_s111
                                 + 0.998684 w_k1_IBM_s111
                                 + 0.855904 w_k1_INTC_s111
                                 + 0.915139 w_k1_KO_s111
                                 + 0.93814 w_k1_MAT_s111
                                 + 0.78419 w_k1_MSFT_s111 + 1.0146 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04848 w_k1_ADM_s111
                                 + 1.14325 w_k1_AXP_s111
                                 + 1.41438 w_k1_DIS_s111
                                 + 1.21701 w_k1_FDX_s111
                                 + 1.15188 w_k1_IBM_s111
                                 + 1.05493 w_k1_INTC_s111
                                 + 1.18249 w_k1_KO_s111 + 1.2296 w_k1_MAT_s111
                                 + 1.12209 w_k1_MSFT_s111
                                 + 1.07283 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.982718 w_k1_ADM_s111
                                 + 0.769654 w_k1_AXP_s111
                                 + 0.958774 w_k1_DIS_s111
                                 + 0.997453 w_k1_FDX_s111
                                 + 0.634978 w_k1_IBM_s111
                                 + 0.702491 w_k1_INTC_s111
                                 + 0.814139 w_k1_KO_s111
                                 + 0.53119 w_k1_MAT_s111
                                 + 0.791304 w_k1_MSFT_s111
                                 + 0.914595 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.818819 w_k1_ADM_s111
                                 + 1.24911 w_k1_AXP_s111
                                 + 1.37253 w_k1_DIS_s111
                                 + 1.28297 w_k1_FDX_s111
                                 + 0.878334 w_k1_IBM_s111
                                 + 0.923367 w_k1_INTC_s111
                                 + 1.0361 w_k1_KO_s111 + 1.06701 w_k1_MAT_s111
                                 + 1.079 w_k1_MSFT_s111 + 1.08957 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.25665 w_k1_ADM_s111
                                 + 1.07722 w_k1_AXP_s111
                                 + 1.21205 w_k1_DIS_s111
                                 + 1.14754 w_k1_FDX_s111
                                 + 1.40855 w_k1_IBM_s111
                                 + 0.943969 w_k1_INTC_s111
                                 + 1.14355 w_k1_KO_s111 + 1.03933 w_k1_MAT_s111
                                 + 0.981767 w_k1_MSFT_s111
                                 + 1.26633 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.994917 w_k1_ADM_s112
                                 + 0.732383 w_k1_AXP_s112
                                 + 1.04698 w_k1_DIS_s112
                                 + 0.964815 w_k1_FDX_s112
                                 + 0.728519 w_k1_IBM_s112
                                 + 0.90838 w_k1_INTC_s112
                                 + 0.882676 w_k1_KO_s112
                                 + 0.674131 w_k1_MAT_s112
                                 + 1.11232 w_k1_MSFT_s112
                                 + 0.867237 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.29988 w_k1_ADM_s112
                                 + 1.66857 w_k1_AXP_s112 + 1.6224 w_k1_DIS_s112
                                 + 1.41168 w_k1_FDX_s112
                                 + 1.48394 w_k1_IBM_s112
                                 + 1.2212 w_k1_INTC_s112 + 1.26893 w_k1_KO_s112
                                 + 1.78258 w_k1_MAT_s112
                                 + 1.20218 w_k1_MSFT_s112
                                 + 1.04849 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.996412 w_k1_ADM_s112
                                 + 0.907415 w_k1_AXP_s112
                                 + 1.21618 w_k1_DIS_s112
                                 + 0.969376 w_k1_FDX_s112
                                 + 1.23588 w_k1_IBM_s112
                                 + 0.955216 w_k1_INTC_s112
                                 + 1.12225 w_k1_KO_s112 + 1.07788 w_k1_MAT_s112
                                 + 1.01743 w_k1_MSFT_s112 + 1.0043 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.810108 w_k1_ADM_s112
                                 + 0.755418 w_k1_AXP_s112
                                 + 0.705472 w_k1_DIS_s112
                                 + 0.775602 w_k1_FDX_s112
                                 + 0.796905 w_k1_IBM_s112
                                 + 0.814877 w_k1_INTC_s112
                                 + 0.88401 w_k1_KO_s112
                                 + 0.946843 w_k1_MAT_s112
                                 + 0.846939 w_k1_MSFT_s112
                                 + 0.769353 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.18793 w_k1_ADM_s112
                                 + 1.35285 w_k1_AXP_s112
                                 + 1.18451 w_k1_DIS_s112
                                 + 1.17869 w_k1_FDX_s112
                                 + 1.31024 w_k1_IBM_s112
                                 + 1.19346 w_k1_INTC_s112
                                 + 1.22105 w_k1_KO_s112 + 1.70766 w_k1_MAT_s112
                                 + 1.23747 w_k1_MSFT_s112
                                 + 1.09725 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.985637 w_k1_ADM_s112
                                 + 1.08628 w_k1_AXP_s112
                                 + 0.992226 w_k1_DIS_s112
                                 + 1.00589 w_k1_FDX_s112
                                 + 0.957626 w_k1_IBM_s112
                                 + 1.09263 w_k1_INTC_s112
                                 + 1.01897 w_k1_KO_s112 + 1.13736 w_k1_MAT_s112
                                 + 1.0094 w_k1_MSFT_s112 + 0.985 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.976606 w_k1_ADM_s112
                                 + 0.816183 w_k1_AXP_s112
                                 + 0.794849 w_k1_DIS_s112
                                 + 0.811859 w_k1_FDX_s112
                                 + 1.23916 w_k1_IBM_s112
                                 + 0.971301 w_k1_INTC_s112
                                 + 1.00766 w_k1_KO_s112
                                 + 0.989486 w_k1_MAT_s112
                                 + 0.822493 w_k1_MSFT_s112
                                 + 0.816048 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.974913 w_k1_ADM_s112
                                 + 0.969742 w_k1_AXP_s112
                                 + 0.822344 w_k1_DIS_s112
                                 + 0.927726 w_k1_FDX_s112
                                 + 0.703715 w_k1_IBM_s112
                                 + 1.0849 w_k1_INTC_s112
                                 + 0.874894 w_k1_KO_s112
                                 + 0.773976 w_k1_MAT_s112
                                 + 1.04672 w_k1_MSFT_s112
                                 + 0.901225 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.697902 w_k1_ADM_s113
                                 + 0.576295 w_k1_AXP_s113
                                 + 0.534146 w_k1_DIS_s113
                                 + 0.726405 w_k1_FDX_s113
                                 + 0.665571 w_k1_IBM_s113
                                 + 0.87737 w_k1_INTC_s113
                                 + 0.71727 w_k1_KO_s113
                                 + 0.527763 w_k1_MAT_s113
                                 + 0.689366 w_k1_MSFT_s113
                                 + 0.704484 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.1161 w_k1_ADM_s113
                                 + 1.01061 w_k1_AXP_s113
                                 + 0.762526 w_k1_DIS_s113
                                 + 0.889689 w_k1_FDX_s113
                                 + 0.990871 w_k1_IBM_s113
                                 + 0.96574 w_k1_INTC_s113
                                 + 0.902914 w_k1_KO_s113
                                 + 0.846295 w_k1_MAT_s113
                                 + 0.74484 w_k1_MSFT_s113
                                 + 1.00896 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.39822 w_k1_ADM_s113
                                 + 1.39008 w_k1_AXP_s113
                                 + 0.908611 w_k1_DIS_s113
                                 + 1.24117 w_k1_FDX_s113
                                 + 0.814221 w_k1_IBM_s113
                                 + 1.28212 w_k1_INTC_s113
                                 + 0.99357 w_k1_KO_s113
                                 + 0.986827 w_k1_MAT_s113
                                 + 1.06863 w_k1_MSFT_s113
                                 + 0.935339 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.06516 w_k1_ADM_s113
                                 + 0.92461 w_k1_AXP_s113
                                 + 1.07858 w_k1_DIS_s113
                                 + 1.06499 w_k1_FDX_s113
                                 + 1.05842 w_k1_IBM_s113
                                 + 0.984524 w_k1_INTC_s113
                                 + 1.06433 w_k1_KO_s113 + 0.69495 w_k1_MAT_s113
                                 + 0.886621 w_k1_MSFT_s113
                                 + 1.06872 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.31251 w_k1_ADM_s113
                                 + 1.53449 w_k1_AXP_s113
                                 + 1.56075 w_k1_DIS_s113
                                 + 1.47552 w_k1_FDX_s113
                                 + 1.18107 w_k1_IBM_s113
                                 + 1.40928 w_k1_INTC_s113
                                 + 1.26356 w_k1_KO_s113 + 1.42326 w_k1_MAT_s113
                                 + 1.26803 w_k1_MSFT_s113
                                 + 1.15352 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.831085 w_k1_ADM_s113
                                 + 0.823246 w_k1_AXP_s113
                                 + 0.671221 w_k1_DIS_s113
                                 + 0.928747 w_k1_FDX_s113
                                 + 0.608079 w_k1_IBM_s113
                                 + 1.01278 w_k1_INTC_s113
                                 + 0.76354 w_k1_KO_s113
                                 + 0.446343 w_k1_MAT_s113
                                 + 0.873683 w_k1_MSFT_s113
                                 + 0.954995 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.919207 w_k1_ADM_s113
                                 + 1.51597 w_k1_AXP_s113
                                 + 0.940654 w_k1_DIS_s113
                                 + 1.02083 w_k1_FDX_s113
                                 + 1.60543 w_k1_IBM_s113
                                 + 0.919755 w_k1_INTC_s113
                                 + 1.06066 w_k1_KO_s113 + 1.6137 w_k1_MAT_s113
                                 + 0.775395 w_k1_MSFT_s113
                                 + 0.947666 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.798435 w_k1_ADM_s113
                                 + 0.847216 w_k1_AXP_s113
                                 + 0.581169 w_k1_DIS_s113
                                 + 0.757034 w_k1_FDX_s113
                                 + 0.957265 w_k1_IBM_s113
                                 + 1.03937 w_k1_INTC_s113
                                 + 0.875866 w_k1_KO_s113
                                 + 0.861079 w_k1_MAT_s113
                                 + 0.81002 w_k1_MSFT_s113
                                 + 0.802383 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.00571 w_k1_ADM_s114
                                 + 0.908523 w_k1_AXP_s114
                                 + 1.02876 w_k1_DIS_s114
                                 + 1.12822 w_k1_FDX_s114
                                 + 0.784156 w_k1_IBM_s114
                                 + 1.26487 w_k1_INTC_s114
                                 + 1.02302 w_k1_KO_s114
                                 + 0.930758 w_k1_MAT_s114
                                 + 1.11415 w_k1_MSFT_s114
                                 + 1.10936 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.10138 w_k1_ADM_s114
                                 + 0.995186 w_k1_AXP_s114
                                 + 1.08255 w_k1_DIS_s114
                                 + 1.09489 w_k1_FDX_s114
                                 + 1.09822 w_k1_IBM_s114
                                 + 1.25879 w_k1_INTC_s114
                                 + 1.07596 w_k1_KO_s114 + 0.83982 w_k1_MAT_s114
                                 + 1.15252 w_k1_MSFT_s114
                                 + 0.964021 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.809813 w_k1_ADM_s114
                                 + 1.14804 w_k1_AXP_s114
                                 + 1.26092 w_k1_DIS_s114 + 1.2846 w_k1_FDX_s114
                                 + 0.769832 w_k1_IBM_s114
                                 + 1.36232 w_k1_INTC_s114
                                 + 1.02333 w_k1_KO_s114
                                 + 0.789565 w_k1_MAT_s114
                                 + 1.27812 w_k1_MSFT_s114
                                 + 0.994448 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.1928 w_k1_ADM_s114
                                 + 1.27109 w_k1_AXP_s114
                                 + 1.33294 w_k1_DIS_s114
                                 + 1.14851 w_k1_FDX_s114
                                 + 0.997665 w_k1_IBM_s114
                                 + 1.36754 w_k1_INTC_s114
                                 + 1.42663 w_k1_KO_s114 + 1.34629 w_k1_MAT_s114
                                 + 1.35142 w_k1_MSFT_s114
                                 + 1.14972 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.11194 w_k1_ADM_s114
                                 + 1.00181 w_k1_AXP_s114 + 1.4171 w_k1_DIS_s114
                                 + 1.09607 w_k1_FDX_s114
                                 + 1.38998 w_k1_IBM_s114
                                 + 1.1858 w_k1_INTC_s114 + 1.21878 w_k1_KO_s114
                                 + 1.17025 w_k1_MAT_s114
                                 + 1.01711 w_k1_MSFT_s114 + 1.0081 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.456615 w_k1_ADM_s114
                                 + 0.761374 w_k1_AXP_s114
                                 + 0.708802 w_k1_DIS_s114
                                 + 0.79996 w_k1_FDX_s114
                                 + 0.890561 w_k1_IBM_s114
                                 + 0.982487 w_k1_INTC_s114
                                 + 0.801167 w_k1_KO_s114
                                 + 0.873719 w_k1_MAT_s114
                                 + 0.825455 w_k1_MSFT_s114
                                 + 0.797829 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.70483 w_k1_ADM_s114
                                 + 0.667135 w_k1_AXP_s114
                                 + 0.626147 w_k1_DIS_s114
                                 + 0.814312 w_k1_FDX_s114
                                 + 0.620895 w_k1_IBM_s114
                                 + 0.862754 w_k1_INTC_s114
                                 + 0.739359 w_k1_KO_s114
                                 + 0.577614 w_k1_MAT_s114
                                 + 0.831101 w_k1_MSFT_s114
                                 + 0.797284 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.808992 w_k1_ADM_s114
                                 + 1.32813 w_k1_AXP_s114
                                 + 1.03219 w_k1_DIS_s114
                                 + 0.926901 w_k1_FDX_s114
                                 + 1.70018 w_k1_IBM_s114
                                 + 1.27757 w_k1_INTC_s114
                                 + 1.34005 w_k1_KO_s114 + 1.79903 w_k1_MAT_s114
                                 + 1.2251 w_k1_MSFT_s114 + 1.07713 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.0467 w_k1_ADM_s115
                                 + 1.2792 w_k1_AXP_s115
                                 + 0.963655 w_k1_DIS_s115
                                 + 0.992835 w_k1_FDX_s115
                                 + 1.20249 w_k1_IBM_s115
                                 + 0.915359 w_k1_INTC_s115
                                 + 0.903381 w_k1_KO_s115
                                 + 1.16994 w_k1_MAT_s115
                                 + 0.896117 w_k1_MSFT_s115
                                 + 1.05813 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.19013 w_k1_ADM_s115
                                 + 1.10674 w_k1_AXP_s115
                                 + 1.28985 w_k1_DIS_s115
                                 + 1.14657 w_k1_FDX_s115
                                 + 1.05517 w_k1_IBM_s115
                                 + 1.02472 w_k1_INTC_s115
                                 + 1.22588 w_k1_KO_s115 + 1.05117 w_k1_MAT_s115
                                 + 1.23867 w_k1_MSFT_s115
                                 + 1.12528 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.844535 w_k1_ADM_s115
                                 + 0.892622 w_k1_AXP_s115
                                 + 1.19317 w_k1_DIS_s115
                                 + 1.11539 w_k1_FDX_s115
                                 + 1.03691 w_k1_IBM_s115
                                 + 0.980257 w_k1_INTC_s115
                                 + 0.935985 w_k1_KO_s115
                                 + 0.704456 w_k1_MAT_s115
                                 + 1.11348 w_k1_MSFT_s115
                                 + 1.02359 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.597473 w_k1_ADM_s115
                                 + 0.517919 w_k1_AXP_s115
                                 + 0.465848 w_k1_DIS_s115
                                 + 0.627553 w_k1_FDX_s115
                                 + 0.811846 w_k1_IBM_s115
                                 + 0.64426 w_k1_INTC_s115
                                 + 0.701504 w_k1_KO_s115
                                 + 0.626699 w_k1_MAT_s115
                                 + 0.702355 w_k1_MSFT_s115
                                 + 0.806744 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.22139 w_k1_ADM_s115
                                 + 1.26069 w_k1_AXP_s115
                                 + 1.06376 w_k1_DIS_s115
                                 + 1.22844 w_k1_FDX_s115
                                 + 0.710004 w_k1_IBM_s115
                                 + 0.889359 w_k1_INTC_s115
                                 + 0.827764 w_k1_KO_s115
                                 + 0.764161 w_k1_MAT_s115
                                 + 0.936108 w_k1_MSFT_s115
                                 + 1.09505 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.08062 w_k1_ADM_s115
                                 + 1.52736 w_k1_AXP_s115
                                 + 1.75563 w_k1_DIS_s115
                                 + 1.32808 w_k1_FDX_s115
                                 + 1.32017 w_k1_IBM_s115
                                 + 1.16285 w_k1_INTC_s115
                                 + 1.34793 w_k1_KO_s115 + 1.49587 w_k1_MAT_s115
                                 + 1.26662 w_k1_MSFT_s115
                                 + 1.43271 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.978973 w_k1_ADM_s115
                                 + 1.11902 w_k1_AXP_s115 + 1.0529 w_k1_DIS_s115
                                 + 1.00864 w_k1_FDX_s115
                                 + 1.55603 w_k1_IBM_s115
                                 + 1.03832 w_k1_INTC_s115
                                 + 1.18843 w_k1_KO_s115 + 1.76562 w_k1_MAT_s115
                                 + 0.994927 w_k1_MSFT_s115
                                 + 1.02091 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.816377 w_k1_ADM_s115
                                 + 0.84869 w_k1_AXP_s115
                                 + 0.823671 w_k1_DIS_s115
                                 + 0.870567 w_k1_FDX_s115
                                 + 0.782863 w_k1_IBM_s115
                                 + 0.771298 w_k1_INTC_s115
                                 + 0.820276 w_k1_KO_s115
                                 + 0.890392 w_k1_MAT_s115
                                 + 0.916829 w_k1_MSFT_s115
                                 + 1.05867 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.826419 w_k1_ADM_s116
                                 + 0.693725 w_k1_AXP_s116
                                 + 0.603461 w_k1_DIS_s116
                                 + 0.735402 w_k1_FDX_s116
                                 + 0.893265 w_k1_IBM_s116
                                 + 0.876801 w_k1_INTC_s116
                                 + 0.77748 w_k1_KO_s116 + 1.05382 w_k1_MAT_s116
                                 + 0.851994 w_k1_MSFT_s116
                                 + 0.803434 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.988816 w_k1_ADM_s116
                                 + 0.668981 w_k1_AXP_s116
                                 + 0.626812 w_k1_DIS_s116
                                 + 0.774314 w_k1_FDX_s116
                                 + 0.615467 w_k1_IBM_s116
                                 + 0.915307 w_k1_INTC_s116
                                 + 0.627232 w_k1_KO_s116
                                 + 0.559319 w_k1_MAT_s116
                                 + 0.902513 w_k1_MSFT_s116
                                 + 0.745596 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.756532 w_k1_ADM_s116
                                 + 0.78071 w_k1_AXP_s116
                                 + 0.956603 w_k1_DIS_s116
                                 + 0.959569 w_k1_FDX_s116
                                 + 0.865034 w_k1_IBM_s116
                                 + 1.07341 w_k1_INTC_s116
                                 + 0.811078 w_k1_KO_s116
                                 + 0.809358 w_k1_MAT_s116
                                 + 1.17156 w_k1_MSFT_s116
                                 + 0.932522 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.933894 w_k1_ADM_s116
                                 + 1.36011 w_k1_AXP_s116
                                 + 1.04701 w_k1_DIS_s116
                                 + 0.976452 w_k1_FDX_s116
                                 + 1.48832 w_k1_IBM_s116
                                 + 1.13106 w_k1_INTC_s116
                                 + 0.977252 w_k1_KO_s116 + 1.4986 w_k1_MAT_s116
                                 + 1.06858 w_k1_MSFT_s116
                                 + 0.956323 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.15729 w_k1_ADM_s116
                                 + 1.08729 w_k1_AXP_s116
                                 + 0.868994 w_k1_DIS_s116
                                 + 1.11626 w_k1_FDX_s116
                                 + 0.812316 w_k1_IBM_s116
                                 + 1.04687 w_k1_INTC_s116
                                 + 0.638663 w_k1_KO_s116
                                 + 0.720575 w_k1_MAT_s116
                                 + 1.05844 w_k1_MSFT_s116
                                 + 0.857473 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.01163 w_k1_ADM_s116
                                 + 1.07026 w_k1_AXP_s116
                                 + 1.18347 w_k1_DIS_s116
                                 + 1.04993 w_k1_FDX_s116
                                 + 1.08553 w_k1_IBM_s116
                                 + 0.898574 w_k1_INTC_s116
                                 + 1.03292 w_k1_KO_s116 + 1.16188 w_k1_MAT_s116
                                 + 1.18281 w_k1_MSFT_s116
                                 + 1.06018 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.19098 w_k1_ADM_s116
                                 + 1.43136 w_k1_AXP_s116
                                 + 1.61544 w_k1_DIS_s116 + 1.4077 w_k1_FDX_s116
                                 + 1.46832 w_k1_IBM_s116
                                 + 1.45393 w_k1_INTC_s116
                                 + 1.21726 w_k1_KO_s116 + 1.57232 w_k1_MAT_s116
                                 + 1.40901 w_k1_MSFT_s116
                                 + 1.17065 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.26117 w_k1_ADM_s116
                                 + 1.43853 w_k1_AXP_s116
                                 + 1.83758 w_k1_DIS_s116
                                 + 1.42051 w_k1_FDX_s116
                                 + 1.21562 w_k1_IBM_s116
                                 + 1.5874 w_k1_INTC_s116 + 1.23151 w_k1_KO_s116
                                 + 1.2124 w_k1_MAT_s116
                                 + 1.55337 w_k1_MSFT_s116
                                 + 1.18105 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11061 w_k1_ADM_s117
                                 + 1.7233 w_k1_AXP_s117 + 1.83205 w_k1_DIS_s117
                                 + 1.44682 w_k1_FDX_s117
                                 + 0.834505 w_k1_IBM_s117
                                 + 1.05233 w_k1_INTC_s117
                                 + 1.24215 w_k1_KO_s117 + 1.08088 w_k1_MAT_s117
                                 + 1.39591 w_k1_MSFT_s117
                                 + 1.06843 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.976223 w_k1_ADM_s117
                                 + 1.14469 w_k1_AXP_s117
                                 + 1.09467 w_k1_DIS_s117
                                 + 1.09156 w_k1_FDX_s117
                                 + 0.624694 w_k1_IBM_s117
                                 + 0.948654 w_k1_INTC_s117
                                 + 0.808069 w_k1_KO_s117
                                 + 0.508169 w_k1_MAT_s117
                                 + 1.00494 w_k1_MSFT_s117
                                 + 0.930077 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.978569 w_k1_ADM_s117
                                 + 0.92827 w_k1_AXP_s117
                                 + 0.901237 w_k1_DIS_s117
                                 + 0.858162 w_k1_FDX_s117
                                 + 0.936363 w_k1_IBM_s117
                                 + 0.989258 w_k1_INTC_s117
                                 + 1.00044 w_k1_KO_s117
                                 + 0.749549 w_k1_MAT_s117
                                 + 0.983019 w_k1_MSFT_s117
                                 + 0.936563 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.05369 w_k1_ADM_s117
                                 + 1.26867 w_k1_AXP_s117
                                 + 1.30534 w_k1_DIS_s117
                                 + 1.15018 w_k1_FDX_s117
                                 + 0.958698 w_k1_IBM_s117
                                 + 1.1196 w_k1_INTC_s117 + 1.10219 w_k1_KO_s117
                                 + 0.852661 w_k1_MAT_s117
                                 + 1.22819 w_k1_MSFT_s117
                                 + 1.14756 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.980822 w_k1_ADM_s117
                                 + 1.05701 w_k1_AXP_s117
                                 + 0.967845 w_k1_DIS_s117
                                 + 0.846956 w_k1_FDX_s117
                                 + 1.18413 w_k1_IBM_s117
                                 + 0.807602 w_k1_INTC_s117
                                 + 1.00259 w_k1_KO_s117 + 1.18837 w_k1_MAT_s117
                                 + 0.902524 w_k1_MSFT_s117
                                 + 1.08089 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.941267 w_k1_ADM_s117
                                 + 0.957513 w_k1_AXP_s117
                                 + 0.641215 w_k1_DIS_s117
                                 + 0.710409 w_k1_FDX_s117
                                 + 0.989845 w_k1_IBM_s117
                                 + 0.672639 w_k1_INTC_s117
                                 + 0.734136 w_k1_KO_s117
                                 + 0.782774 w_k1_MAT_s117
                                 + 0.655401 w_k1_MSFT_s117
                                 + 0.814899 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.869605 w_k1_ADM_s117
                                 + 0.834711 w_k1_AXP_s117
                                 + 0.823982 w_k1_DIS_s117
                                 + 0.808445 w_k1_FDX_s117
                                 + 0.741629 w_k1_IBM_s117
                                 + 0.856294 w_k1_INTC_s117
                                 + 0.826708 w_k1_KO_s117
                                 + 0.749109 w_k1_MAT_s117
                                 + 0.900525 w_k1_MSFT_s117
                                 + 0.835993 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.25066 w_k1_ADM_s117
                                 + 1.3984 w_k1_AXP_s117 + 1.40524 w_k1_DIS_s117
                                 + 1.10395 w_k1_FDX_s117
                                 + 1.41005 w_k1_IBM_s117
                                 + 1.09337 w_k1_INTC_s117
                                 + 1.37268 w_k1_KO_s117 + 1.31329 w_k1_MAT_s117
                                 + 1.17172 w_k1_MSFT_s117
                                 + 1.01321 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0337612383641992 w_k1_ADM_s000
                                 + 0.0139209862377148 w_k1_ADM_s001
                                 + 0.0346184266694806 w_k1_ADM_s002
                                 + 0.0157190406028558 w_k1_ADM_s003
                                 + 0.0500581683146333 w_k1_ADM_s004
                                 + 0.0460257140753153 w_k1_ADM_s005
                                 + 0.0339581261000643 w_k1_ADM_s006
                                 + 0.0332013321005969 w_k1_ADM_s007
                                 + 0.105141416249071 w_k1_ADM_s010
                                 + 0.0590003919808912 w_k1_ADM_s011
                                 + 0.0211252218245469 w_k1_ADM_s012
                                 + 0.0370184210400158 w_k1_ADM_s013
                                 + 0.0575274985742225 w_k1_ADM_s014
                                 + 0.0438856082335579 w_k1_ADM_s015
                                 + 0.0424132471245586 w_k1_ADM_s016
                                 + 0.0601563572636243 w_k1_ADM_s017
                                 + 0.0195883437859303 w_k1_ADM_s100
                                 + 0.0203194184770226 w_k1_ADM_s101
                                 + 0.0146852082506216 w_k1_ADM_s102
                                 + 0.0356034519600258 w_k1_ADM_s103
                                 + 0.0350772771470776 w_k1_ADM_s104
                                 + 0.0197030262238274 w_k1_ADM_s105
                                 + 0.045712604683575 w_k1_ADM_s106
                                 + 0.0286057826667989 w_k1_ADM_s107
                                 + 0.0219741381746215 w_k1_ADM_s110
                                 + 0.0119627760205182 w_k1_ADM_s111
                                 + 0.0248665553080712 w_k1_ADM_s112
                                 + 0.0138867295153328 w_k1_ADM_s113
                                 + 0.00621178532788633 w_k1_ADM_s114
                                 + 0.0167639495185972 w_k1_ADM_s115
                                 + 0.0112186723514012 w_k1_ADM_s116
                                 + 0.0231261760708831 w_k1_ADM_s117
                                 + 0.0352768831946046 w_k1_AXP_s000
                                 + 0.0153518273885205 w_k1_AXP_s001
                                 + 0.0375844721046458 w_k1_AXP_s002
                                 + 0.0169384961427396 w_k1_AXP_s003
                                 + 0.0437791256383968 w_k1_AXP_s004
                                 + 0.0454450614258882 w_k1_AXP_s005
                                 + 0.0405544085731959 w_k1_AXP_s006
                                 + 0.0310211029705105 w_k1_AXP_s007
                                 + 0.0961033457472703 w_k1_AXP_s010
                                 + 0.0540261011434567 w_k1_AXP_s011
                                 + 0.0213897796915194 w_k1_AXP_s012
                                 + 0.0342549518458079 w_k1_AXP_s013
                                 + 0.070827275793927 w_k1_AXP_s014
                                 + 0.0442253549979431 w_k1_AXP_s015
                                 + 0.0358613082177376 w_k1_AXP_s016
                                 + 0.0622974440339745 w_k1_AXP_s017
                                 + 0.0214418540563239 w_k1_AXP_s100
                                 + 0.0182153315900213 w_k1_AXP_s101
                                 + 0.0132791672726553 w_k1_AXP_s102
                                 + 0.0330933852107604 w_k1_AXP_s103
                                 + 0.0285632226391634 w_k1_AXP_s104
                                 + 0.0192860928915074 w_k1_AXP_s105
                                 + 0.0427088483572274 w_k1_AXP_s106
                                 + 0.0296793324817306 w_k1_AXP_s107
                                 + 0.0232344239144165 w_k1_AXP_s110
                                 + 0.0136080590052369 w_k1_AXP_s111
                                 + 0.0255753738145821 w_k1_AXP_s112
                                 + 0.0144153865796061 w_k1_AXP_s113
                                 + 0.00674410480915333 w_k1_AXP_s114
                                 + 0.0187432036070428 w_k1_AXP_s115
                                 + 0.0117701468369951 w_k1_AXP_s116
                                 + 0.025788663823604 w_k1_AXP_s117
                                 + 0.0354894984984259 w_k1_DIS_s000
                                 + 0.0154099903243585 w_k1_DIS_s001
                                 + 0.0389877596652935 w_k1_DIS_s002
                                 + 0.0183567179445481 w_k1_DIS_s003
                                 + 0.0452731427996363 w_k1_DIS_s004
                                 + 0.0414981127187514 w_k1_DIS_s005
                                 + 0.0393114227882558 w_k1_DIS_s006
                                 + 0.030797602216274 w_k1_DIS_s007
                                 + 0.0963653783407608 w_k1_DIS_s010
                                 + 0.055271697056447 w_k1_DIS_s011
                                 + 0.0222731756811744 w_k1_DIS_s012
                                 + 0.0333532202544412 w_k1_DIS_s013
                                 + 0.0685817332736019 w_k1_DIS_s014
                                 + 0.0474051811091068 w_k1_DIS_s015
                                 + 0.0399666884006767 w_k1_DIS_s016
                                 + 0.061507405487348 w_k1_DIS_s017
                                 + 0.0205977406264752 w_k1_DIS_s100
                                 + 0.0186597659301519 w_k1_DIS_s101
                                 + 0.0133831042472502 w_k1_DIS_s102
                                 + 0.0331330708765701 w_k1_DIS_s103
                                 + 0.0272632187853965 w_k1_DIS_s104
                                 + 0.0211041284558883 w_k1_DIS_s105
                                 + 0.042321861152877 w_k1_DIS_s106
                                 + 0.0312197608262738 w_k1_DIS_s107
                                 + 0.0247619393748258 w_k1_DIS_s110
                                 + 0.0133712088438569 w_k1_DIS_s111
                                 + 0.0248309159166644 w_k1_DIS_s112
                                 + 0.0126708493875417 w_k1_DIS_s113
                                 + 0.00717775896885347 w_k1_DIS_s114
                                 + 0.0185208344272546 w_k1_DIS_s115
                                 + 0.0120880647394877 w_k1_DIS_s116
                                 + 0.0245185812690938 w_k1_DIS_s117
                                 + 0.0332139390224258 w_k1_FDX_s000
                                 + 0.0159543728273983 w_k1_FDX_s001
                                 + 0.0365109660897545 w_k1_FDX_s002
                                 + 0.0178149546837299 w_k1_FDX_s003
                                 + 0.0480270887357544 w_k1_FDX_s004
                                 + 0.0444352112318618 w_k1_FDX_s005
                                 + 0.0392624245805903 w_k1_FDX_s006
                                 + 0.031306544664657 w_k1_FDX_s007
                                 + 0.0966197567653937 w_k1_FDX_s010
                                 + 0.0539014172495278 w_k1_FDX_s011
                                 + 0.0207419777532712 w_k1_FDX_s012
                                 + 0.0348205419546763 w_k1_FDX_s013
                                 + 0.0643570149157345 w_k1_FDX_s014
                                 + 0.0453218096570542 w_k1_FDX_s015
                                 + 0.0377848201701559 w_k1_FDX_s016
                                 + 0.0583299038067927 w_k1_FDX_s017
                                 + 0.0201656245907475 w_k1_FDX_s100
                                 + 0.0179649520117073 w_k1_FDX_s101
                                 + 0.0126680911585861 w_k1_FDX_s102
                                 + 0.0339538722806111 w_k1_FDX_s103
                                 + 0.0283203515532123 w_k1_FDX_s104
                                 + 0.0197623908446553 w_k1_FDX_s105
                                 + 0.0429459719915738 w_k1_FDX_s106
                                 + 0.0320392499952238 w_k1_FDX_s107
                                 + 0.0236636138754292 w_k1_FDX_s110
                                 + 0.0137975354145166 w_k1_FDX_s111
                                 + 0.0244739701367079 w_k1_FDX_s112
                                 + 0.0138801166103389 w_k1_FDX_s113
                                 + 0.0070896437532392 w_k1_FDX_s114
                                 + 0.0177720734365356 w_k1_FDX_s115
                                 + 0.0117636011521162 w_k1_FDX_s116
                                 + 0.0217904968469706 w_k1_FDX_s117
                                 + 0.0352246462255915 w_k1_IBM_s000
                                 + 0.0159189437619583 w_k1_IBM_s001
                                 + 0.0363513436696781 w_k1_IBM_s002
                                 + 0.0167280007195476 w_k1_IBM_s003
                                 + 0.0482960011858811 w_k1_IBM_s004
                                 + 0.043994142074699 w_k1_IBM_s005
                                 + 0.0373397952953173 w_k1_IBM_s006
                                 + 0.0300272625914193 w_k1_IBM_s007
                                 + 0.0958627696624769 w_k1_IBM_s010
                                 + 0.0562082915352701 w_k1_IBM_s011
                                 + 0.0206828476786587 w_k1_IBM_s012
                                 + 0.0334051930233411 w_k1_IBM_s013
                                 + 0.0649470602305927 w_k1_IBM_s014
                                 + 0.0434249571155829 w_k1_IBM_s015
                                 + 0.0371756820342845 w_k1_IBM_s016
                                 + 0.0643995790942372 w_k1_IBM_s017
                                 + 0.0196194005841909 w_k1_IBM_s100
                                 + 0.0189142734768767 w_k1_IBM_s101
                                 + 0.013619530886821 w_k1_IBM_s102
                                 + 0.0335416886438208 w_k1_IBM_s103
                                 + 0.0282876459516493 w_k1_IBM_s104
                                 + 0.0196838326479466 w_k1_IBM_s105
                                 + 0.042606774241605 w_k1_IBM_s106
                                 + 0.030596168595765 w_k1_IBM_s107
                                 + 0.0235121627934616 w_k1_IBM_s110
                                 + 0.0131159000263496 w_k1_IBM_s111
                                 + 0.0251059919183233 w_k1_IBM_s112
                                 + 0.0130608234375036 w_k1_IBM_s113
                                 + 0.00718147692460797 w_k1_IBM_s114
                                 + 0.0180588668599313 w_k1_IBM_s115
                                 + 0.0120231053273934 w_k1_IBM_s116
                                 + 0.0229914966356307 w_k1_IBM_s117
                                 + 0.0317816084965823 w_k1_INTC_s000
                                 + 0.0154705059991061 w_k1_INTC_s001
                                 + 0.0435127359647728 w_k1_INTC_s002
                                 + 0.0161832363097652 w_k1_INTC_s003
                                 + 0.0407877748969763 w_k1_INTC_s004
                                 + 0.0510194485417222 w_k1_INTC_s005
                                 + 0.040868637349051 w_k1_INTC_s006
                                 + 0.0282467272548486 w_k1_INTC_s007
                                 + 0.0954403050437617 w_k1_INTC_s010
                                 + 0.0486773958287982 w_k1_INTC_s011
                                 + 0.0188246812240264 w_k1_INTC_s012
                                 + 0.0342532429300012 w_k1_INTC_s013
                                 + 0.0580946484527281 w_k1_INTC_s014
                                 + 0.0457821035590793 w_k1_INTC_s015
                                 + 0.0356346022323751 w_k1_INTC_s016
                                 + 0.0676755935075731 w_k1_INTC_s017
                                 + 0.0223511866756407 w_k1_INTC_s100
                                 + 0.0175424244925919 w_k1_INTC_s101
                                 + 0.0146440310724573 w_k1_INTC_s102
                                 + 0.0333894687367696 w_k1_INTC_s103
                                 + 0.0265642713406255 w_k1_INTC_s104
                                 + 0.0175701803619915 w_k1_INTC_s105
                                 + 0.0420192760791412 w_k1_INTC_s106
                                 + 0.0302600224190379 w_k1_INTC_s107
                                 + 0.0252587363254103 w_k1_INTC_s110
                                 + 0.0113902181785493 w_k1_INTC_s111
                                 + 0.0252889834515433 w_k1_INTC_s112
                                 + 0.0142619415867263 w_k1_INTC_s113
                                 + 0.00816582854734674 w_k1_INTC_s114
                                 + 0.0158414396549344 w_k1_INTC_s115
                                 + 0.0123011299277153 w_k1_INTC_s116
                                 + 0.0209200465396971 w_k1_INTC_s117
                                 + 0.0340946507595614 w_k1_KO_s000
                                 + 0.0145734220153036 w_k1_KO_s001
                                 + 0.0344184252373159 w_k1_KO_s002
                                 + 0.0167132714316347 w_k1_KO_s003
                                 + 0.0463292798048472 w_k1_KO_s004
                                 + 0.041491208689305 w_k1_KO_s005
                                 + 0.0362930468375716 w_k1_KO_s006
                                 + 0.0302544716983082 w_k1_KO_s007
                                 + 0.103484657778425 w_k1_KO_s010
                                 + 0.0571159305836742 w_k1_KO_s011
                                 + 0.021711333856691 w_k1_KO_s012
                                 + 0.0329356534769546 w_k1_KO_s013
                                 + 0.0626611480339026 w_k1_KO_s014
                                 + 0.0437646433279475 w_k1_KO_s015
                                 + 0.039787038552407 w_k1_KO_s016
                                 + 0.0588725441343911 w_k1_KO_s017
                                 + 0.0202781400586523 w_k1_KO_s100
                                 + 0.0197161063326645 w_k1_KO_s101
                                 + 0.0134286922176748 w_k1_KO_s102
                                 + 0.028866845691727 w_k1_KO_s103
                                 + 0.0280035796715022 w_k1_KO_s104
                                 + 0.0195427216657559 w_k1_KO_s105
                                 + 0.0420789783918502 w_k1_KO_s106
                                 + 0.028468927688397 w_k1_KO_s107
                                 + 0.0267318579892207 w_k1_KO_s110
                                 + 0.0125977727910537 w_k1_KO_s111
                                 + 0.0252164496026851 w_k1_KO_s112
                                 + 0.013013276632402 w_k1_KO_s113
                                 + 0.00727499464112425 w_k1_KO_s114
                                 + 0.0169667164991402 w_k1_KO_s115
                                 + 0.0104761693535093 w_k1_KO_s116
                                 + 0.0230513916243622 w_k1_KO_s117
                                 + 0.0402411653073654 w_k1_MAT_s000
                                 + 0.016775375074984 w_k1_MAT_s001
                                 + 0.0347461369042525 w_k1_MAT_s002
                                 + 0.0163886514662672 w_k1_MAT_s003
                                 + 0.0447992351531081 w_k1_MAT_s004
                                 + 0.039685435003029 w_k1_MAT_s005
                                 + 0.0359286081215358 w_k1_MAT_s006
                                 + 0.0321943752982239 w_k1_MAT_s007
                                 + 0.0814083579784753 w_k1_MAT_s010
                                 + 0.0539309426986449 w_k1_MAT_s011
                                 + 0.0255160787622867 w_k1_MAT_s012
                                 + 0.0351478498612311 w_k1_MAT_s013
                                 + 0.0682886186349171 w_k1_MAT_s014
                                 + 0.0469870618846212 w_k1_MAT_s015
                                 + 0.0337181533173415 w_k1_MAT_s016
                                 + 0.0604447397634459 w_k1_MAT_s017
                                 + 0.0244132721639398 w_k1_MAT_s100
                                 + 0.0208510973969237 w_k1_MAT_s101
                                 + 0.0140819215057821 w_k1_MAT_s102
                                 + 0.0312177894163846 w_k1_MAT_s103
                                 + 0.0276232462694963 w_k1_MAT_s104
                                 + 0.0196080116247101 w_k1_MAT_s105
                                 + 0.0381079624871267 w_k1_MAT_s106
                                 + 0.0321109931757893 w_k1_MAT_s107
                                 + 0.0233222791076027 w_k1_MAT_s110
                                 + 0.0128870196230661 w_k1_MAT_s111
                                 + 0.0286491468560945 w_k1_MAT_s112
                                 + 0.0122961942548066 w_k1_MAT_s113
                                 + 0.00678343346002093 w_k1_MAT_s114
                                 + 0.0185278152111757 w_k1_MAT_s115
                                 + 0.0122702503447626 w_k1_MAT_s116
                                 + 0.0215491209932297 w_k1_MAT_s117
                                 + 0.03373362963812 w_k1_MSFT_s000
                                 + 0.0162815863936058 w_k1_MSFT_s001
                                 + 0.0390693887086255 w_k1_MSFT_s002
                                 + 0.0169881883228724 w_k1_MSFT_s003
                                 + 0.0446594255885875 w_k1_MSFT_s004
                                 + 0.0435291968212831 w_k1_MSFT_s005
                                 + 0.0367189202667047 w_k1_MSFT_s006
                                 + 0.0298248946707126 w_k1_MSFT_s007
                                 + 0.0953153481833559 w_k1_MSFT_s010
                                 + 0.0481666767428795 w_k1_MSFT_s011
                                 + 0.0217849977629443 w_k1_MSFT_s012
                                 + 0.0361720164023549 w_k1_MSFT_s013
                                 + 0.0658965417587671 w_k1_MSFT_s014
                                 + 0.0414265829085899 w_k1_MSFT_s015
                                 + 0.03549130473651 w_k1_MSFT_s016
                                 + 0.0570050773268942 w_k1_MSFT_s017
                                 + 0.0200994099960659 w_k1_MSFT_s100
                                 + 0.0201882697987537 w_k1_MSFT_s101
                                 + 0.0141240255568254 w_k1_MSFT_s102
                                 + 0.032353191007908 w_k1_MSFT_s103
                                 + 0.025946817297328 w_k1_MSFT_s104
                                 + 0.0214104852538661 w_k1_MSFT_s105
                                 + 0.0405869912005225 w_k1_MSFT_s106
                                 + 0.0312216266857436 w_k1_MSFT_s107
                                 + 0.0224594034090949 w_k1_MSFT_s110
                                 + 0.0118666872831943 w_k1_MSFT_s111
                                 + 0.0255188007470327 w_k1_MSFT_s112
                                 + 0.0121319894724261 w_k1_MSFT_s113
                                 + 0.00742801691177644 w_k1_MSFT_s114
                                 + 0.0172484072472799 w_k1_MSFT_s115
                                 + 0.0130968796744339 w_k1_MSFT_s116
                                 + 0.0226719160449052 w_k1_MSFT_s117
                                 + 0.0350968824793167 w_k1_PG_s000
                                 + 0.0146939636771158 w_k1_PG_s001
                                 + 0.0339505850734862 w_k1_PG_s002
                                 + 0.0175157669241404 w_k1_PG_s003
                                 + 0.0453651139394946 w_k1_PG_s004
                                 + 0.0432638875371858 w_k1_PG_s005
                                 + 0.0407018682018076 w_k1_PG_s006
                                 + 0.0296973057025047 w_k1_PG_s007
                                 + 0.0962800285886185 w_k1_PG_s010
                                 + 0.0566798128795361 w_k1_PG_s011
                                 + 0.0220920701913582 w_k1_PG_s012
                                 + 0.0339696139686699 w_k1_PG_s013
                                 + 0.0656848272783338 w_k1_PG_s014
                                 + 0.0460324686603502 w_k1_PG_s015
                                 + 0.0384665551933885 w_k1_PG_s016
                                 + 0.0587546007805672 w_k1_PG_s017
                                 + 0.0208094698712377 w_k1_PG_s100
                                 + 0.0182287880375923 w_k1_PG_s101
                                 + 0.0130028017474088 w_k1_PG_s102
                                 + 0.0334040894409783 w_k1_PG_s103
                                 + 0.0286966474727318 w_k1_PG_s104
                                 + 0.0188134202414545 w_k1_PG_s105
                                 + 0.0402880689881144 w_k1_PG_s106
                                 + 0.030528609629315 w_k1_PG_s107
                                 + 0.0238986202361733 w_k1_PG_s110
                                 + 0.0136713291462394 w_k1_PG_s111
                                 + 0.0230621543510777 w_k1_PG_s112
                                 + 0.0128196840830094 w_k1_PG_s113
                                 + 0.00666113766843492 w_k1_PG_s114
                                 + 0.0186954981522202 w_k1_PG_s115
                                 + 0.0111202242756065 w_k1_PG_s116
                                 + 0.0219805831275368 w_k1_PG_s117 >= 1.11
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
