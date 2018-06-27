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
 w_k1_ADM_s0_return_constraint:  0.997623 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.01062 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.3705 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.984687 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01416 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.22209 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.11706 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.08411 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.734904 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.958051 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.864963 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.993414 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.812887 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.953855 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.01563 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.03798 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.822876 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.739173 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.935317 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.20774 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.820832 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.22365 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.993462 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.28377 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.822692 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.847229 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.95898 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.743807 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.06991 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.570217 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.916126 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.877137 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.07126 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.449 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.0899 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0093 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.836962 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.11088 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.968734 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.16004 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.43933 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.920034 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.803627 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.32344 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.47155 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.14523 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.957519 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.11512 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.98499 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.1386 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.759845 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.19467 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.825521 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.08863 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.859774 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.572966 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.788599 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.40042 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.04088 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.31505 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.02785 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.3253 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.700318 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.843438 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.0117 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.890448 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.38397 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.552626 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.04802 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.814237 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.22987 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.58902 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.14917 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.752412 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.700295 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.18901 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.00338 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.14017 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.54531 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.878776 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.782999 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.34404 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.71741 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.48874 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.877886 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.29029 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.905851 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.869512 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.779821 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.22823 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.812737 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.13892 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.05564 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.526068 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.722666 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.58055 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.913106 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.52233 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.996703 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.61846 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.647994 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.794481 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.972711 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.899811 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.27561 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.490956 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.805123 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.919087 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.38428 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.98564 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.26448 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.00242 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.721868 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.09978 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.00677 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.99823 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.39467 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.923984 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.861915 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.29354 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.36858 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.30676 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.802079 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.05538 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.867022 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.01361 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.841 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.06133 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.01383 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.11117 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.966562 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.641312 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.853776 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.26616 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.918151 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.38878 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.916059 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.27656 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.798139 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.910333 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.12995 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.906503 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.22108 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.646723 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.914376 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.82696 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.21889 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.44572 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.24436 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.97353 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.807958 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.993356 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.912367 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.34711 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.25137 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.942073 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.922952 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.1667 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.48565 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.767153 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.62435 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.23446 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.16931 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.798971 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.566422 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.98601 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.569448 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.927823 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.839873 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.72659 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.912132 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.49623 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.35273 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.900471 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.47135 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.3716 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.594349 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.786062 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.749038 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.01635 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.962132 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.667999 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.988976 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.17525 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.01929 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.41066 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.673836 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.724874 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.664892 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.55117 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.935697 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.08147 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.4048 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.921956 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.891249 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.04375 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.13942 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.11553 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.903571 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.963064 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.957226 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.940264 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.856297 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.04554 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.967811 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.12212 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.08557 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.719735 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.9534 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.14198 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.01002 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.52483 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.838728 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.18675 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.734587 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.841113 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.968163 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.888656 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.05939 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.633954 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.962526 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.962081 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.3674 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.58202 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.31205 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.988502 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.875289 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.04127 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.973939 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.1569 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.26366 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.887783 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.853947 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.14844 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.35221 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.997082 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.16571 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.29113 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.03498 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.883118 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.796939 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.1083 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.798698 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.01148 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.01577 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.65989 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.87146 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.44303 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.08269 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.21146 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.30804 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.49319 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.778783 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.962339 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.925066 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.03303 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.09539 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.596671 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02119 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.1682 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.18042 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.69896 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.00864 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.958098 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.721533 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.28977 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.897161 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.25328 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.33841 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.986822 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.86803 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.39753 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.52223 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.623835 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.45086 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.24318 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.1625 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.845077 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.61833 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.08022 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.598269 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.910188 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.831723 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.523952 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.806372 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.68696 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.61117 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.955481 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.69132 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.4646 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.784982 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.851373 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.801791 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.19378 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.983259 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.532243 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.99384 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.15058 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.961871 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.96186 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.841974 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.619641 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.739406 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.47718 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.998983 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.02338 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.30104 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.888523 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.935596 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16825 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.19615 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.17328 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.833825 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.13387 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.951499 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.955071 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.929446 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.1558 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.06327 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.14152 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.14172 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.728468 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.921501 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.17023 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.960831 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.33814 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.959558 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.18268 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.755751 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.808811 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.940652 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.808948 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.978851 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.652409 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.809021 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.950024 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.18791 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.59047 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.29766 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.0544 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.885782 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.977023 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.01025 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.08399 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.20142 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.904168 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.897419 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.15754 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.18982 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.03755 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.992994 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0614 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.886359 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.968252 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.765859 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.00801 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.908613 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.01371 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.984674 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.812171 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.881507 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.17742 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04271 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.14315 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.0032 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.23147 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.809375 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.915738 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02741 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.954476 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.16305 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.689865 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.920481 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.936035 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.11098 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.35148 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.984491 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.930119 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.799015 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.08477 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.455000307291581 z_k1_s0000
                                 - 0.417999256540931 z_k1_s0001
                                 - 0.235000209743977 z_k1_s0002
                                 - 0.202999815591796 z_k1_s0003
                                 - 0.256000366818898 z_k1_s0004
                                 - 0.289000613650918 z_k1_s0005
                                 - 0.225999910226274 z_k1_s0006
                                 - 0.23999947314317 z_k1_s0007
                                 - 0.140000171708403 z_k1_s0010
                                 - 0.0509999787009453 z_k1_s0011
                                 - 0.124999540561272 z_k1_s0012
                                 - 0.13700040667886 z_k1_s0013
                                 - 0.161999953591231 z_k1_s0014
                                 - 0.154999899855826 z_k1_s0015
                                 - 0.0740000133601809 z_k1_s0016
                                 - 0.0590000143128457 z_k1_s0017
                                 - 0.0739999271067687 z_k1_s0020
                                 - 0.0869999697768506 z_k1_s0021
                                 - 0.130000426700819 z_k1_s0022
                                 - 0.199000098165361 z_k1_s0023
                                 - 0.232999846133439 z_k1_s0024
                                 - 0.114999908574022 z_k1_s0025
                                 - 0.244999464475251 z_k1_s0026
                                 - 0.160999648083737 z_k1_s0027
                                 - 0.202999573412217 z_k1_s0030
                                 - 0.0479999938116883 z_k1_s0031
                                 - 0.0599999922646104 z_k1_s0032
                                 - 0.119999984529221 z_k1_s0033
                                 - 0.0779999899439935 z_k1_s0034
                                 - 0.0959998915234451 z_k1_s0035
                                 - 0.224000291454317 z_k1_s0036
                                 - 0.131999982982143 z_k1_s0037
                                 - 0.48499834894087 z_k1_s0040
                                 - 0.590000127792784 z_k1_s0041
                                 - 0.365999857073282 z_k1_s0042
                                 - 0.49900055965329 z_k1_s0043
                                 - 0.893997383863949 z_k1_s0044
                                 - 0.347999975026989 z_k1_s0045
                                 - 0.371999958721963 z_k1_s0046
                                 - 0.674998748484097 z_k1_s0047
                                 - 0.496997598209677 z_k1_s0050
                                 - 0.195999444242308 z_k1_s0051
                                 - 0.609998379417123 z_k1_s0052
                                 - 0.336000268869579 z_k1_s0053
                                 - 0.445000319735095 z_k1_s0054
                                 - 0.261999278913645 z_k1_s0055
                                 - 0.21999955070421 z_k1_s0056
                                 - 0.487997481936648 z_k1_s0057
                                 - 0.341999569731329 z_k1_s0060
                                 - 0.0970000509544904 z_k1_s0061
                                 - 0.230000929429497 z_k1_s0062
                                 - 0.135000301947349 z_k1_s0063
                                 - 0.225001250246805 z_k1_s0064
                                 - 0.428999814119702 z_k1_s0065
                                 - 0.340000594459719 z_k1_s0066
                                 - 0.44300160503564 z_k1_s0067
                                 - 0.176999923018109 z_k1_s0070
                                 - 0.0489999646683057 z_k1_s0071
                                 - 0.0949998850753382 z_k1_s0072
                                 - 0.16600024325957 z_k1_s0073
                                 - 0.106000392032956 z_k1_s0074
                                 - 0.136999677534083 z_k1_s0075
                                 - 0.235999897357479 z_k1_s0076
                                 - 0.0679999003234193 z_k1_s0077
                                 - 0.393000159239631 z_k1_s0100
                                 - 0.506000662209325 z_k1_s0101
                                 - 0.342000205394035 z_k1_s0102
                                 - 0.30800008259696 z_k1_s0103
                                 - 0.781001723170958 z_k1_s0104
                                 - 0.449000090493619 z_k1_s0105
                                 - 1.26799850920815 z_k1_s0106
                                 - 0.559999312539876 z_k1_s0107
                                 - 1.61400446690577 z_k1_s0110
                                 - 0.330000908843174 z_k1_s0111
                                 - 0.995001302735182 z_k1_s0112
                                 - 0.436001470090535 z_k1_s0113
                                 - 0.663001924212475 z_k1_s0114
                                 - 0.739001420198572 z_k1_s0115
                                 - 0.340001063763026 z_k1_s0116
                                 - 0.888003908654881 z_k1_s0117
                                 - 0.471999519649065 z_k1_s0120
                                 - 0.218999932525305 z_k1_s0121
                                 - 0.402001144887523 z_k1_s0122
                                 - 0.5039994870829 z_k1_s0123
                                 - 0.52500024271134 z_k1_s0124
                                 - 0.168000139827628 z_k1_s0125
                                 - 0.351999641772184 z_k1_s0126
                                 - 0.466001079755193 z_k1_s0127
                                 - 1.13400480227257 z_k1_s0130
                                 - 1.42600687140375 z_k1_s0131
                                 - 1.32900476947528 z_k1_s0132
                                 - 0.739999686409668 z_k1_s0133
                                 - 0.580001907214812 z_k1_s0134
                                 - 0.340000812807657 z_k1_s0135
                                 - 0.865001178416492 z_k1_s0136
                                 - 0.96200328034496 z_k1_s0137
                                 - 1.45999960858774 z_k1_s0140
                                 - 1.15199715367598 z_k1_s0141
                                 - 0.854002168757657 z_k1_s0142
                                 - 0.820000317768188 z_k1_s0143
                                 - 0.834002926762928 z_k1_s0144
                                 - 0.738999463191498 z_k1_s0145
                                 - 1.64099745803768 z_k1_s0146
                                 - 0.349000116101348 z_k1_s0147
                                 - 2.18500521762554 z_k1_s0150
                                 - 1.55899780291958 z_k1_s0151
                                 - 1.48900242889932 z_k1_s0152
                                 - 0.707000927270687 z_k1_s0153
                                 - 1.60099899813724 z_k1_s0154
                                 - 0.876001387621437 z_k1_s0155
                                 - 0.518000512298094 z_k1_s0156
                                 - 0.992001521508682 z_k1_s0157
                                 - 1.69099667470823 z_k1_s0160
                                 - 1.18000245067526 z_k1_s0161
                                 - 0.688000468938743 z_k1_s0162
                                 - 1.80999606153839 z_k1_s0163
                                 - 1.46300000468236 z_k1_s0164
                                 - 1.85900209494384 z_k1_s0165
                                 - 1.30000632227212 z_k1_s0166
                                 - 0.695000872882029 z_k1_s0167
                                 - 0.26600012350809 z_k1_s0170
                                 - 0.410000034567359 z_k1_s0171
                                 - 1.25200028421252 z_k1_s0172
                                 - 0.451000086214094 z_k1_s0173
                                 - 0.791000291184585 z_k1_s0174
                                 - 0.455999798671016 z_k1_s0175
                                 - 0.707000302908346 z_k1_s0176
                                 - 0.48600096481252 z_k1_s0177
                                 - 0.43599919267529 z_k1_s1000
                                 - 0.544999781092847 z_k1_s1001
                                 - 0.47799933240513 z_k1_s1002
                                 - 0.560997576469735 z_k1_s1003
                                 - 0.223999796927333 z_k1_s1004
                                 - 0.366997834192361 z_k1_s1005
                                 - 0.27899952833618 z_k1_s1006
                                 - 0.270999682349255 z_k1_s1007
                                 - 0.680999484409924 z_k1_s1010
                                 - 0.251999865604502 z_k1_s1011
                                 - 0.211999842762558 z_k1_s1012
                                 - 0.173999960212585 z_k1_s1013
                                 - 0.246999654024448 z_k1_s1014
                                 - 0.495999726640615 z_k1_s1015
                                 - 0.385998968075897 z_k1_s1016
                                 - 0.335000539175971 z_k1_s1017
                                 - 0.250999973769057 z_k1_s1020
                                 - 0.11199999725352 z_k1_s1021
                                 - 0.276000287678223 z_k1_s1022
                                 - 0.170000035983009 z_k1_s1023
                                 - 0.217999499448977 z_k1_s1024
                                 - 0.1960001825686 z_k1_s1025
                                 - 0.202000677626664 z_k1_s1026
                                 - 0.0739999098514617 z_k1_s1027
                                 - 0.251999943423183 z_k1_s1030
                                 - 0.0889999347931198 z_k1_s1031
                                 - 0.370000369184716 z_k1_s1032
                                 - 0.161999939686338 z_k1_s1033
                                 - 0.127000040655316 z_k1_s1034
                                 - 0.177999534386325 z_k1_s1035
                                 - 0.206000459210885 z_k1_s1036
                                 - 0.291999349173044 z_k1_s1037
                                 - 0.238000561006678 z_k1_s1040
                                 - 0.164000403974392 z_k1_s1041
                                 - 0.351000937194148 z_k1_s1042
                                 - 0.288001100775489 z_k1_s1043
                                 - 0.206000381187041 z_k1_s1044
                                 - 0.247000611580952 z_k1_s1045
                                 - 0.358001494130947 z_k1_s1046
                                 - 0.736000512642905 z_k1_s1047
                                 - 0.249000121440055 z_k1_s1050
                                 - 0.570998662791489 z_k1_s1051
                                 - 0.17800004708245 z_k1_s1052
                                 - 0.212999948272822 z_k1_s1053
                                 - 0.240999869225119 z_k1_s1054
                                 - 0.587000266421335 z_k1_s1055
                                 - 0.277001141592327 z_k1_s1056
                                 - 0.431999330006863 z_k1_s1057
                                 - 0.191999762780927 z_k1_s1060
                                 - 0.290000394220835 z_k1_s1061
                                 - 0.299999739470147 z_k1_s1062
                                 - 0.266999257148673 z_k1_s1063
                                 - 0.217999822428301 z_k1_s1064
                                 - 0.113999936238639 z_k1_s1065
                                 - 0.262000465523599 z_k1_s1066
                                 - 0.118999961263128 z_k1_s1067
                                 - 0.0749999734819538 z_k1_s1070
                                 - 0.0759999168084178 z_k1_s1071
                                 - 0.0669999868705383 z_k1_s1072
                                 - 0.0819999287669646 z_k1_s1073
                                 - 0.0769999481348225 z_k1_s1074
                                 - 0.234999441710442 z_k1_s1075
                                 - 0.20899985922309 z_k1_s1076
                                 - 0.0590000002591228 z_k1_s1077
                                 - 0.16899957081741 z_k1_s1100
                                 - 0.22699993845931 z_k1_s1101
                                 - 0.179999883853142 z_k1_s1102
                                 - 0.141999764639583 z_k1_s1103
                                 - 0.162999613889004 z_k1_s1104
                                 - 0.0690000926771506 z_k1_s1105
                                 - 0.178000682210984 z_k1_s1106
                                 - 0.0480000082275368 z_k1_s1107
                                 - 0.182000127036188 z_k1_s1110
                                 - 0.14300022922141 z_k1_s1111
                                 - 0.225999717542335 z_k1_s1112
                                 - 0.267000539194622 z_k1_s1113
                                 - 0.385000694557713 z_k1_s1114
                                 - 0.128000144646381 z_k1_s1115
                                 - 0.508001146512726 z_k1_s1116
                                 - 0.174000256389667 z_k1_s1117
                                 - 0.199000237040454 z_k1_s1120
                                 - 0.225000241010504 z_k1_s1121
                                 - 0.117000196965619 z_k1_s1122
                                 - 0.189001062130707 z_k1_s1123
                                 - 0.133000309624353 z_k1_s1124
                                 - 0.22100092924739 z_k1_s1125
                                 - 0.273000937187491 z_k1_s1126
                                 - 0.434001443964629 z_k1_s1127
                                 - 0.172000558387934 z_k1_s1130
                                 - 0.264999891795058 z_k1_s1131
                                 - 0.210000652403864 z_k1_s1132
                                 - 0.0900000632587305 z_k1_s1133
                                 - 0.118000066112552 z_k1_s1134
                                 - 0.188999703763198 z_k1_s1135
                                 - 0.135999473004107 z_k1_s1136
                                 - 0.0819999903290443 z_k1_s1137
                                 - 0.0740003083531745 z_k1_s1140
                                 - 0.023000036698901 z_k1_s1141
                                 - 0.0599999824752817 z_k1_s1142
                                 - 0.0839998712652911 z_k1_s1143
                                 - 0.10500036008213 z_k1_s1144
                                 - 0.0450000650065388 z_k1_s1145
                                 - 0.101999918107927 z_k1_s1146
                                 - 0.0280000265551659 z_k1_s1147
                                 - 0.12399967042084 z_k1_s1150
                                 - 0.339999304282495 z_k1_s1151
                                 - 0.355998881473108 z_k1_s1152
                                 - 0.221999199008736 z_k1_s1153
                                 - 0.118999756498873 z_k1_s1154
                                 - 0.177999624936153 z_k1_s1155
                                 - 0.08699986531925 z_k1_s1156
                                 - 0.415998216935904 z_k1_s1157
                                 - 0.185999795303695 z_k1_s1160
                                 - 0.296999128651284 z_k1_s1161
                                 - 0.266000408316826 z_k1_s1162
                                 - 0.632998559367717 z_k1_s1163
                                 - 0.48499893283781 z_k1_s1164
                                 - 0.184000044120759 z_k1_s1165
                                 - 0.238000026257298 z_k1_s1166
                                 - 0.288000248328071 z_k1_s1167
                                 - 0.0649998913795384 z_k1_s1170
                                 - 0.0939998032812421 z_k1_s1171
                                 - 0.129999782759077 z_k1_s1172
                                 - 0.211000041683715 z_k1_s1173
                                 - 0.320999941310493 z_k1_s1174
                                 - 0.208999944890659 z_k1_s1175
                                 - 0.180999945785701 z_k1_s1176
                                 - 0.144999559508519 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.967741 w_k1_ADM_s000
                                 + 1.29239 w_k1_AXP_s000
                                 + 0.878985 w_k1_DIS_s000
                                 + 0.97663 w_k1_FDX_s000
                                 + 1.07681 w_k1_IBM_s000
                                 + 1.30676 w_k1_INTC_s000
                                 + 1.19121 w_k1_KO_s000 + 1.49493 w_k1_MAT_s000
                                 + 1.0531 w_k1_MSFT_s000 + 1.06653 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.865763 w_k1_ADM_s000
                                 + 0.714556 w_k1_AXP_s000
                                 + 0.705858 w_k1_DIS_s000
                                 + 0.788772 w_k1_FDX_s000
                                 + 0.884518 w_k1_IBM_s000
                                 + 0.764498 w_k1_INTC_s000
                                 + 0.84424 w_k1_KO_s000
                                 + 0.906436 w_k1_MAT_s000
                                 + 0.913057 w_k1_MSFT_s000
                                 + 0.881554 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.779827 w_k1_ADM_s000
                                 + 0.859787 w_k1_AXP_s000
                                 + 0.696703 w_k1_DIS_s000
                                 + 0.807591 w_k1_FDX_s000
                                 + 0.56961 w_k1_IBM_s000
                                 + 0.903055 w_k1_INTC_s000
                                 + 0.696763 w_k1_KO_s000
                                 + 0.568682 w_k1_MAT_s000
                                 + 0.849733 w_k1_MSFT_s000
                                 + 0.805445 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.22669 w_k1_ADM_s000
                                 + 1.43903 w_k1_AXP_s000
                                 + 0.891622 w_k1_DIS_s000
                                 + 1.18423 w_k1_FDX_s000
                                 + 0.744568 w_k1_IBM_s000
                                 + 1.25489 w_k1_INTC_s000
                                 + 0.869127 w_k1_KO_s000
                                 + 0.996336 w_k1_MAT_s000
                                 + 1.13649 w_k1_MSFT_s000
                                 + 0.964844 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.837307 w_k1_ADM_s000
                                 + 0.954841 w_k1_AXP_s000
                                 + 0.851392 w_k1_DIS_s000
                                 + 0.742435 w_k1_FDX_s000
                                 + 1.58517 w_k1_IBM_s000
                                 + 0.850467 w_k1_INTC_s000
                                 + 1.20773 w_k1_KO_s000 + 1.53093 w_k1_MAT_s000
                                 + 1.08338 w_k1_MSFT_s000
                                 + 1.12326 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.13637 w_k1_ADM_s000
                                 + 1.05664 w_k1_AXP_s000
                                 + 1.23263 w_k1_DIS_s000
                                 + 1.07692 w_k1_FDX_s000
                                 + 0.80788 w_k1_IBM_s000
                                 + 1.35719 w_k1_INTC_s000 + 1.0714 w_k1_KO_s000
                                 + 1.03343 w_k1_MAT_s000
                                 + 1.36045 w_k1_MSFT_s000
                                 + 1.04936 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.835874 w_k1_ADM_s000
                                 + 1.12423 w_k1_AXP_s000 + 1.4777 w_k1_DIS_s000
                                 + 1.14367 w_k1_FDX_s000
                                 + 1.39446 w_k1_IBM_s000
                                 + 1.38898 w_k1_INTC_s000 + 1.4643 w_k1_KO_s000
                                 + 1.56012 w_k1_MAT_s000
                                 + 1.24431 w_k1_MSFT_s000
                                 + 1.22826 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.43549 w_k1_ADM_s000
                                 + 1.41922 w_k1_AXP_s000
                                 + 1.76412 w_k1_DIS_s000
                                 + 1.42297 w_k1_FDX_s000
                                 + 1.06911 w_k1_IBM_s000
                                 + 1.17904 w_k1_INTC_s000
                                 + 1.48341 w_k1_KO_s000 + 1.35517 w_k1_MAT_s000
                                 + 1.54877 w_k1_MSFT_s000
                                 + 1.22566 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.13138 w_k1_ADM_s001
                                 + 0.846184 w_k1_AXP_s001
                                 + 1.01339 w_k1_DIS_s001
                                 + 0.904817 w_k1_FDX_s001
                                 + 0.973804 w_k1_IBM_s001
                                 + 0.969587 w_k1_INTC_s001
                                 + 1.15706 w_k1_KO_s001 + 1.33941 w_k1_MAT_s001
                                 + 1.05229 w_k1_MSFT_s001
                                 + 1.14656 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.29313 w_k1_ADM_s001
                                 + 1.22321 w_k1_AXP_s001
                                 + 1.66688 w_k1_DIS_s001
                                 + 1.23564 w_k1_FDX_s001
                                 + 1.39362 w_k1_IBM_s001
                                 + 1.27197 w_k1_INTC_s001
                                 + 1.52441 w_k1_KO_s001 + 1.81831 w_k1_MAT_s001
                                 + 1.44065 w_k1_MSFT_s001
                                 + 1.31583 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.03373 w_k1_ADM_s001
                                 + 1.15773 w_k1_AXP_s001
                                 + 0.964926 w_k1_DIS_s001
                                 + 1.02171 w_k1_FDX_s001
                                 + 0.833314 w_k1_IBM_s001
                                 + 1.16788 w_k1_INTC_s001
                                 + 1.04116 w_k1_KO_s001 + 1.01006 w_k1_MAT_s001
                                 + 1.13153 w_k1_MSFT_s001
                                 + 1.02796 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.04777 w_k1_ADM_s001
                                 + 1.31401 w_k1_AXP_s001
                                 + 1.08072 w_k1_DIS_s001 + 1.2134 w_k1_FDX_s001
                                 + 0.911025 w_k1_IBM_s001
                                 + 1.20235 w_k1_INTC_s001
                                 + 0.994063 w_k1_KO_s001
                                 + 1.49457 w_k1_MAT_s001
                                 + 1.24419 w_k1_MSFT_s001
                                 + 1.12928 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.09241 w_k1_ADM_s001
                                 + 1.44346 w_k1_AXP_s001
                                 + 1.21026 w_k1_DIS_s001
                                 + 1.12987 w_k1_FDX_s001
                                 + 1.38453 w_k1_IBM_s001
                                 + 1.25575 w_k1_INTC_s001
                                 + 1.48959 w_k1_KO_s001 + 2.04862 w_k1_MAT_s001
                                 + 1.26324 w_k1_MSFT_s001
                                 + 1.25953 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.952859 w_k1_ADM_s001
                                 + 0.873172 w_k1_AXP_s001
                                 + 0.738498 w_k1_DIS_s001
                                 + 0.850124 w_k1_FDX_s001
                                 + 1.04116 w_k1_IBM_s001
                                 + 0.806244 w_k1_INTC_s001
                                 + 1.00252 w_k1_KO_s001 + 1.36832 w_k1_MAT_s001
                                 + 0.82151 w_k1_MSFT_s001
                                 + 0.943374 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.58337 w_k1_ADM_s001
                                 + 1.8663 w_k1_AXP_s001 + 1.55975 w_k1_DIS_s001
                                 + 1.39089 w_k1_FDX_s001
                                 + 1.29955 w_k1_IBM_s001
                                 + 1.52296 w_k1_INTC_s001
                                 + 1.34052 w_k1_KO_s001 + 2.16988 w_k1_MAT_s001
                                 + 1.45854 w_k1_MSFT_s001
                                 + 1.23029 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.688084 w_k1_ADM_s001
                                 + 0.673522 w_k1_AXP_s001
                                 + 0.576086 w_k1_DIS_s001
                                 + 0.705405 w_k1_FDX_s001
                                 + 0.780747 w_k1_IBM_s001
                                 + 0.725348 w_k1_INTC_s001
                                 + 0.696952 w_k1_KO_s001
                                 + 0.791506 w_k1_MAT_s001
                                 + 0.78967 w_k1_MSFT_s001
                                 + 0.728479 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.895999 w_k1_ADM_s002
                                 + 0.932588 w_k1_AXP_s002
                                 + 0.736624 w_k1_DIS_s002
                                 + 0.808377 w_k1_FDX_s002
                                 + 1.00701 w_k1_IBM_s002
                                 + 0.829831 w_k1_INTC_s002
                                 + 0.831756 w_k1_KO_s002
                                 + 1.39075 w_k1_MAT_s002
                                 + 0.781027 w_k1_MSFT_s002
                                 + 0.884006 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.799651 w_k1_ADM_s002
                                 + 0.728366 w_k1_AXP_s002
                                 + 0.7972 w_k1_DIS_s002 + 0.81236 w_k1_FDX_s002
                                 + 0.606165 w_k1_IBM_s002
                                 + 0.838624 w_k1_INTC_s002
                                 + 0.69674 w_k1_KO_s002
                                 + 0.859049 w_k1_MAT_s002
                                 + 0.925912 w_k1_MSFT_s002
                                 + 0.865081 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.815641 w_k1_ADM_s002
                                 + 1.042 w_k1_AXP_s002 + 1.5648 w_k1_DIS_s002
                                 + 1.20369 w_k1_FDX_s002
                                 + 0.984684 w_k1_IBM_s002
                                 + 1.27653 w_k1_INTC_s002
                                 + 1.11912 w_k1_KO_s002
                                 + 0.969778 w_k1_MAT_s002
                                 + 1.2343 w_k1_MSFT_s002 + 0.95512 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.09382 w_k1_ADM_s002
                                 + 1.05579 w_k1_AXP_s002
                                 + 1.06598 w_k1_DIS_s002
                                 + 1.05131 w_k1_FDX_s002
                                 + 0.811184 w_k1_IBM_s002
                                 + 0.977301 w_k1_INTC_s002
                                 + 0.968283 w_k1_KO_s002
                                 + 1.12947 w_k1_MAT_s002
                                 + 1.10998 w_k1_MSFT_s002
                                 + 1.05999 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.853578 w_k1_ADM_s002
                                 + 1.45987 w_k1_AXP_s002
                                 + 1.68059 w_k1_DIS_s002
                                 + 1.13671 w_k1_FDX_s002
                                 + 1.53264 w_k1_IBM_s002
                                 + 1.30678 w_k1_INTC_s002
                                 + 1.37468 w_k1_KO_s002 + 2.05905 w_k1_MAT_s002
                                 + 1.34688 w_k1_MSFT_s002
                                 + 1.13368 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.694197 w_k1_ADM_s002
                                 + 0.505347 w_k1_AXP_s002
                                 + 0.581711 w_k1_DIS_s002
                                 + 0.676039 w_k1_FDX_s002
                                 + 0.623083 w_k1_IBM_s002
                                 + 0.768829 w_k1_INTC_s002
                                 + 0.570465 w_k1_KO_s002
                                 + 0.610156 w_k1_MAT_s002
                                 + 0.85617 w_k1_MSFT_s002
                                 + 0.729487 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.0401 w_k1_ADM_s002
                                 + 0.951366 w_k1_AXP_s002
                                 + 0.944173 w_k1_DIS_s002
                                 + 0.903596 w_k1_FDX_s002
                                 + 1.15295 w_k1_IBM_s002
                                 + 0.868741 w_k1_INTC_s002
                                 + 1.1798 w_k1_KO_s002 + 1.28553 w_k1_MAT_s002
                                 + 1.05717 w_k1_MSFT_s002
                                 + 1.11716 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.86216 w_k1_ADM_s002
                                 + 0.708556 w_k1_AXP_s002
                                 + 0.850079 w_k1_DIS_s002
                                 + 0.801929 w_k1_FDX_s002
                                 + 0.830327 w_k1_IBM_s002
                                 + 0.877047 w_k1_INTC_s002
                                 + 0.835825 w_k1_KO_s002
                                 + 0.742877 w_k1_MAT_s002
                                 + 0.94509 w_k1_MSFT_s002
                                 + 0.844559 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.21568 w_k1_ADM_s003
                                 + 0.91284 w_k1_AXP_s003
                                 + 1.10871 w_k1_DIS_s003
                                 + 1.06715 w_k1_FDX_s003
                                 + 0.976176 w_k1_IBM_s003
                                 + 0.989911 w_k1_INTC_s003
                                 + 1.02092 w_k1_KO_s003 + 1.38397 w_k1_MAT_s003
                                 + 1.12354 w_k1_MSFT_s003
                                 + 1.06289 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.840012 w_k1_ADM_s003
                                 + 1.11654 w_k1_AXP_s003
                                 + 1.33852 w_k1_DIS_s003
                                 + 1.07722 w_k1_FDX_s003
                                 + 1.85208 w_k1_IBM_s003
                                 + 1.21972 w_k1_INTC_s003
                                 + 1.50797 w_k1_KO_s003 + 1.76291 w_k1_MAT_s003
                                 + 1.02857 w_k1_MSFT_s003
                                 + 1.15283 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.73601 w_k1_ADM_s003
                                 + 1.3475 w_k1_AXP_s003 + 1.81883 w_k1_DIS_s003
                                 + 1.35255 w_k1_FDX_s003
                                 + 1.25649 w_k1_IBM_s003
                                 + 1.18789 w_k1_INTC_s003
                                 + 1.44827 w_k1_KO_s003 + 1.3261 w_k1_MAT_s003
                                 + 1.22724 w_k1_MSFT_s003
                                 + 1.49035 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.93597 w_k1_ADM_s003
                                 + 0.907963 w_k1_AXP_s003
                                 + 0.78103 w_k1_DIS_s003
                                 + 0.860665 w_k1_FDX_s003
                                 + 1.15483 w_k1_IBM_s003
                                 + 0.940517 w_k1_INTC_s003
                                 + 1.0019 w_k1_KO_s003 + 1.30036 w_k1_MAT_s003
                                 + 0.955684 w_k1_MSFT_s003
                                 + 0.936985 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.3673 w_k1_ADM_s003
                                 + 1.36236 w_k1_AXP_s003
                                 + 1.08515 w_k1_DIS_s003
                                 + 1.15267 w_k1_FDX_s003
                                 + 0.909818 w_k1_IBM_s003
                                 + 1.10956 w_k1_INTC_s003 + 1.0584 w_k1_KO_s003
                                 + 1.20982 w_k1_MAT_s003
                                 + 1.11549 w_k1_MSFT_s003
                                 + 1.12356 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.945901 w_k1_ADM_s003
                                 + 1.1599 w_k1_AXP_s003 + 1.36683 w_k1_DIS_s003
                                 + 1.23026 w_k1_FDX_s003
                                 + 1.14588 w_k1_IBM_s003
                                 + 1.28367 w_k1_INTC_s003
                                 + 1.04046 w_k1_KO_s003 + 1.23966 w_k1_MAT_s003
                                 + 1.08244 w_k1_MSFT_s003
                                 + 1.04283 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.841147 w_k1_ADM_s003
                                 + 0.835191 w_k1_AXP_s003
                                 + 0.818534 w_k1_DIS_s003
                                 + 0.896121 w_k1_FDX_s003
                                 + 0.75169 w_k1_IBM_s003
                                 + 0.898809 w_k1_INTC_s003
                                 + 0.793684 w_k1_KO_s003
                                 + 0.715317 w_k1_MAT_s003
                                 + 0.876963 w_k1_MSFT_s003
                                 + 0.843559 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.15633 w_k1_ADM_s003
                                 + 1.18663 w_k1_AXP_s003
                                 + 1.00557 w_k1_DIS_s003
                                 + 1.09672 w_k1_FDX_s003
                                 + 1.26568 w_k1_IBM_s003
                                 + 1.26586 w_k1_INTC_s003
                                 + 1.02411 w_k1_KO_s003 + 1.15997 w_k1_MAT_s003
                                 + 1.06853 w_k1_MSFT_s003
                                 + 1.10626 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.0792 w_k1_ADM_s004
                                 + 1.07584 w_k1_AXP_s004
                                 + 0.873127 w_k1_DIS_s004
                                 + 0.947078 w_k1_FDX_s004
                                 + 1.02537 w_k1_IBM_s004
                                 + 0.848458 w_k1_INTC_s004
                                 + 0.940365 w_k1_KO_s004 + 1.0327 w_k1_MAT_s004
                                 + 0.756478 w_k1_MSFT_s004
                                 + 1.09285 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.40742 w_k1_ADM_s004
                                 + 1.2846 w_k1_AXP_s004 + 1.41136 w_k1_DIS_s004
                                 + 1.22182 w_k1_FDX_s004
                                 + 1.50659 w_k1_IBM_s004
                                 + 1.09155 w_k1_INTC_s004
                                 + 1.22052 w_k1_KO_s004 + 1.27091 w_k1_MAT_s004
                                 + 1.14222 w_k1_MSFT_s004
                                 + 1.26214 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.0521 w_k1_ADM_s004
                                 + 0.846478 w_k1_AXP_s004
                                 + 1.04535 w_k1_DIS_s004
                                 + 1.09504 w_k1_FDX_s004
                                 + 0.773995 w_k1_IBM_s004
                                 + 1.15131 w_k1_INTC_s004
                                 + 1.01571 w_k1_KO_s004
                                 + 0.508219 w_k1_MAT_s004
                                 + 1.03846 w_k1_MSFT_s004
                                 + 1.03229 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.5635 w_k1_ADM_s004
                                 + 1.28109 w_k1_AXP_s004
                                 + 1.51201 w_k1_DIS_s004
                                 + 1.35335 w_k1_FDX_s004
                                 + 0.923786 w_k1_IBM_s004
                                 + 1.09896 w_k1_INTC_s004
                                 + 1.16952 w_k1_KO_s004
                                 + 0.992735 w_k1_MAT_s004
                                 + 1.07069 w_k1_MSFT_s004
                                 + 1.17893 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.038 w_k1_ADM_s004
                                 + 0.976831 w_k1_AXP_s004
                                 + 1.15359 w_k1_DIS_s004
                                 + 1.04504 w_k1_FDX_s004
                                 + 1.03882 w_k1_IBM_s004
                                 + 0.949794 w_k1_INTC_s004
                                 + 1.01674 w_k1_KO_s004 + 1.13884 w_k1_MAT_s004
                                 + 1.0635 w_k1_MSFT_s004 + 1.16755 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.765402 w_k1_ADM_s004
                                 + 0.640411 w_k1_AXP_s004
                                 + 0.679765 w_k1_DIS_s004
                                 + 0.799734 w_k1_FDX_s004
                                 + 0.679119 w_k1_IBM_s004
                                 + 0.688769 w_k1_INTC_s004
                                 + 0.778612 w_k1_KO_s004
                                 + 0.645183 w_k1_MAT_s004
                                 + 0.76798 w_k1_MSFT_s004
                                 + 0.938642 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.10522 w_k1_ADM_s004
                                 + 0.75769 w_k1_AXP_s004
                                 + 0.899391 w_k1_DIS_s004
                                 + 0.94813 w_k1_FDX_s004
                                 + 0.622466 w_k1_IBM_s004
                                 + 0.809215 w_k1_INTC_s004
                                 + 0.741994 w_k1_KO_s004
                                 + 0.585554 w_k1_MAT_s004
                                 + 0.953434 w_k1_MSFT_s004
                                 + 0.946598 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.799146 w_k1_ADM_s004
                                 + 0.672536 w_k1_AXP_s004
                                 + 0.723355 w_k1_DIS_s004
                                 + 0.778566 w_k1_FDX_s004
                                 + 0.861091 w_k1_IBM_s004
                                 + 0.756029 w_k1_INTC_s004
                                 + 0.824394 w_k1_KO_s004
                                 + 0.794619 w_k1_MAT_s004
                                 + 0.845112 w_k1_MSFT_s004
                                 + 0.86765 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.778297 w_k1_ADM_s005
                                 + 0.763102 w_k1_AXP_s005
                                 + 0.797497 w_k1_DIS_s005
                                 + 0.856892 w_k1_FDX_s005
                                 + 0.756819 w_k1_IBM_s005
                                 + 0.88172 w_k1_INTC_s005
                                 + 0.84762 w_k1_KO_s005
                                 + 0.979029 w_k1_MAT_s005
                                 + 0.93723 w_k1_MSFT_s005
                                 + 0.816249 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.00833 w_k1_ADM_s005
                                 + 1.19223 w_k1_AXP_s005
                                 + 1.84986 w_k1_DIS_s005
                                 + 1.24312 w_k1_FDX_s005
                                 + 1.52688 w_k1_IBM_s005
                                 + 1.41683 w_k1_INTC_s005
                                 + 1.44303 w_k1_KO_s005 + 1.3337 w_k1_MAT_s005
                                 + 1.3449 w_k1_MSFT_s005 + 1.26261 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.16284 w_k1_ADM_s005
                                 + 0.987272 w_k1_AXP_s005
                                 + 0.941075 w_k1_DIS_s005
                                 + 1.04367 w_k1_FDX_s005
                                 + 0.924613 w_k1_IBM_s005
                                 + 0.957384 w_k1_INTC_s005
                                 + 0.985538 w_k1_KO_s005
                                 + 1.11132 w_k1_MAT_s005
                                 + 1.09491 w_k1_MSFT_s005
                                 + 1.02463 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.27295 w_k1_ADM_s005
                                 + 1.32206 w_k1_AXP_s005
                                 + 1.42243 w_k1_DIS_s005
                                 + 1.42614 w_k1_FDX_s005
                                 + 0.939786 w_k1_IBM_s005
                                 + 1.09174 w_k1_INTC_s005
                                 + 1.13122 w_k1_KO_s005 + 1.11637 w_k1_MAT_s005
                                 + 1.25149 w_k1_MSFT_s005
                                 + 1.15911 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.04347 w_k1_ADM_s005
                                 + 1.07795 w_k1_AXP_s005
                                 + 1.21074 w_k1_DIS_s005 + 1.0455 w_k1_FDX_s005
                                 + 1.41403 w_k1_IBM_s005
                                 + 1.19499 w_k1_INTC_s005
                                 + 1.24034 w_k1_KO_s005 + 1.63683 w_k1_MAT_s005
                                 + 1.04781 w_k1_MSFT_s005
                                 + 1.06687 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.10843 w_k1_ADM_s005
                                 + 0.845506 w_k1_AXP_s005
                                 + 1.28519 w_k1_DIS_s005
                                 + 1.10396 w_k1_FDX_s005
                                 + 0.952321 w_k1_IBM_s005
                                 + 1.04269 w_k1_INTC_s005
                                 + 1.19018 w_k1_KO_s005
                                 + 0.960718 w_k1_MAT_s005
                                 + 1.25243 w_k1_MSFT_s005
                                 + 1.01812 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.871497 w_k1_ADM_s005
                                 + 0.845275 w_k1_AXP_s005
                                 + 1.09763 w_k1_DIS_s005
                                 + 0.883119 w_k1_FDX_s005
                                 + 1.2447 w_k1_IBM_s005
                                 + 0.921823 w_k1_INTC_s005
                                 + 1.1988 w_k1_KO_s005 + 1.48373 w_k1_MAT_s005
                                 + 0.97605 w_k1_MSFT_s005
                                 + 0.987847 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.790149 w_k1_ADM_s005
                                 + 0.743599 w_k1_AXP_s005
                                 + 0.775738 w_k1_DIS_s005
                                 + 0.726699 w_k1_FDX_s005
                                 + 1.07026 w_k1_IBM_s005
                                 + 0.730021 w_k1_INTC_s005
                                 + 0.883962 w_k1_KO_s005
                                 + 1.07125 w_k1_MAT_s005
                                 + 0.728203 w_k1_MSFT_s005
                                 + 0.885173 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.863871 w_k1_ADM_s006
                                 + 0.845182 w_k1_AXP_s006
                                 + 0.833953 w_k1_DIS_s006
                                 + 0.845549 w_k1_FDX_s006
                                 + 1.46207 w_k1_IBM_s006
                                 + 0.941039 w_k1_INTC_s006
                                 + 1.01984 w_k1_KO_s006 + 1.34927 w_k1_MAT_s006
                                 + 0.878138 w_k1_MSFT_s006
                                 + 0.966892 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.27008 w_k1_ADM_s006
                                 + 0.989869 w_k1_AXP_s006
                                 + 1.3743 w_k1_DIS_s006 + 1.08994 w_k1_FDX_s006
                                 + 1.94545 w_k1_IBM_s006
                                 + 1.01504 w_k1_INTC_s006
                                 + 1.43034 w_k1_KO_s006 + 1.70838 w_k1_MAT_s006
                                 + 1.0585 w_k1_MSFT_s006 + 1.30086 w_k1_PG_s006
                                 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.11348 w_k1_ADM_s006
                                 + 0.988244 w_k1_AXP_s006
                                 + 1.4199 w_k1_DIS_s006 + 1.21687 w_k1_FDX_s006
                                 + 1.34892 w_k1_IBM_s006
                                 + 1.15872 w_k1_INTC_s006 + 1.4024 w_k1_KO_s006
                                 + 1.12383 w_k1_MAT_s006
                                 + 1.22271 w_k1_MSFT_s006
                                 + 1.27653 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.00054 w_k1_ADM_s006
                                 + 1.26002 w_k1_AXP_s006 + 1.084 w_k1_DIS_s006
                                 + 1.14469 w_k1_FDX_s006 + 1.2844 w_k1_IBM_s006
                                 + 0.9328 w_k1_INTC_s006
                                 + 0.964281 w_k1_KO_s006
                                 + 1.31508 w_k1_MAT_s006
                                 + 0.954052 w_k1_MSFT_s006
                                 + 1.12769 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.896336 w_k1_ADM_s006
                                 + 0.939765 w_k1_AXP_s006
                                 + 0.733819 w_k1_DIS_s006
                                 + 0.885463 w_k1_FDX_s006
                                 + 0.915094 w_k1_IBM_s006
                                 + 0.733787 w_k1_INTC_s006
                                 + 0.824211 w_k1_KO_s006
                                 + 1.07292 w_k1_MAT_s006
                                 + 0.880167 w_k1_MSFT_s006
                                 + 0.984747 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.79348 w_k1_ADM_s006
                                 + 0.641628 w_k1_AXP_s006
                                 + 0.658169 w_k1_DIS_s006
                                 + 0.784194 w_k1_FDX_s006
                                 + 0.841161 w_k1_IBM_s006
                                 + 0.76985 w_k1_INTC_s006
                                 + 0.717173 w_k1_KO_s006
                                 + 0.745659 w_k1_MAT_s006
                                 + 0.784927 w_k1_MSFT_s006
                                 + 0.949631 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.22488 w_k1_ADM_s006
                                 + 1.30354 w_k1_AXP_s006
                                 + 1.51098 w_k1_DIS_s006
                                 + 1.43167 w_k1_FDX_s006
                                 + 0.954453 w_k1_IBM_s006
                                 + 1.18503 w_k1_INTC_s006
                                 + 1.10249 w_k1_KO_s006 + 1.15076 w_k1_MAT_s006
                                 + 1.41546 w_k1_MSFT_s006 + 1.2613 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.09396 w_k1_ADM_s006
                                 + 1.01081 w_k1_AXP_s006
                                 + 1.05331 w_k1_DIS_s006
                                 + 1.03491 w_k1_FDX_s006
                                 + 1.00068 w_k1_IBM_s006
                                 + 1.03651 w_k1_INTC_s006
                                 + 0.92632 w_k1_KO_s006
                                 + 0.732154 w_k1_MAT_s006
                                 + 1.00243 w_k1_MSFT_s006
                                 + 1.15792 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.13145 w_k1_ADM_s007
                                 + 0.952254 w_k1_AXP_s007
                                 + 0.998605 w_k1_DIS_s007
                                 + 1.07758 w_k1_FDX_s007
                                 + 0.792089 w_k1_IBM_s007
                                 + 0.985506 w_k1_INTC_s007
                                 + 1.00066 w_k1_KO_s007
                                 + 0.593093 w_k1_MAT_s007
                                 + 0.983286 w_k1_MSFT_s007
                                 + 1.01996 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.52053 w_k1_ADM_s007
                                 + 1.6843 w_k1_AXP_s007 + 1.71555 w_k1_DIS_s007
                                 + 1.33394 w_k1_FDX_s007 + 1.5383 w_k1_IBM_s007
                                 + 1.18427 w_k1_INTC_s007
                                 + 1.36038 w_k1_KO_s007 + 1.77206 w_k1_MAT_s007
                                 + 1.20757 w_k1_MSFT_s007
                                 + 1.35104 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.25993 w_k1_ADM_s007
                                 + 1.23905 w_k1_AXP_s007
                                 + 1.56309 w_k1_DIS_s007
                                 + 1.34477 w_k1_FDX_s007
                                 + 1.08056 w_k1_IBM_s007
                                 + 1.30765 w_k1_INTC_s007 + 1.1679 w_k1_KO_s007
                                 + 0.840969 w_k1_MAT_s007
                                 + 1.33267 w_k1_MSFT_s007
                                 + 1.25349 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.21117 w_k1_ADM_s007
                                 + 1.223 w_k1_AXP_s007 + 1.18568 w_k1_DIS_s007
                                 + 1.24044 w_k1_FDX_s007
                                 + 1.07941 w_k1_IBM_s007
                                 + 0.986583 w_k1_INTC_s007
                                 + 1.08318 w_k1_KO_s007 + 1.28086 w_k1_MAT_s007
                                 + 1.06295 w_k1_MSFT_s007
                                 + 1.18211 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.692204 w_k1_ADM_s007
                                 + 0.64516 w_k1_AXP_s007
                                 + 0.846059 w_k1_DIS_s007
                                 + 0.804761 w_k1_FDX_s007
                                 + 0.82616 w_k1_IBM_s007
                                 + 0.864887 w_k1_INTC_s007
                                 + 0.917582 w_k1_KO_s007 + 0.7555 w_k1_MAT_s007
                                 + 0.852354 w_k1_MSFT_s007
                                 + 0.945622 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.940653 w_k1_ADM_s007
                                 + 0.962736 w_k1_AXP_s007
                                 + 0.71987 w_k1_DIS_s007
                                 + 0.949693 w_k1_FDX_s007
                                 + 0.853845 w_k1_IBM_s007
                                 + 1.01693 w_k1_INTC_s007
                                 + 0.696035 w_k1_KO_s007
                                 + 0.795927 w_k1_MAT_s007
                                 + 0.795743 w_k1_MSFT_s007
                                 + 0.85051 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.0505 w_k1_ADM_s007
                                 + 1.06273 w_k1_AXP_s007
                                 + 1.11451 w_k1_DIS_s007
                                 + 0.989825 w_k1_FDX_s007
                                 + 1.23115 w_k1_IBM_s007
                                 + 1.00562 w_k1_INTC_s007
                                 + 1.08256 w_k1_KO_s007 + 1.15169 w_k1_MAT_s007
                                 + 1.04829 w_k1_MSFT_s007
                                 + 1.05827 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.815676 w_k1_ADM_s007
                                 + 0.689549 w_k1_AXP_s007
                                 + 0.638313 w_k1_DIS_s007
                                 + 0.760651 w_k1_FDX_s007
                                 + 0.829518 w_k1_IBM_s007
                                 + 0.798598 w_k1_INTC_s007
                                 + 0.739454 w_k1_KO_s007
                                 + 0.889624 w_k1_MAT_s007
                                 + 0.710723 w_k1_MSFT_s007
                                 + 0.903011 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.759679 w_k1_ADM_s010
                                 + 0.665345 w_k1_AXP_s010
                                 + 0.628698 w_k1_DIS_s010
                                 + 0.70645 w_k1_FDX_s010
                                 + 0.768614 w_k1_IBM_s010
                                 + 0.973069 w_k1_INTC_s010
                                 + 0.751361 w_k1_KO_s010
                                 + 0.586079 w_k1_MAT_s010
                                 + 0.766311 w_k1_MSFT_s010
                                 + 0.884722 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.923465 w_k1_ADM_s010
                                 + 1.03699 w_k1_AXP_s010
                                 + 1.02268 w_k1_DIS_s010
                                 + 0.934154 w_k1_FDX_s010
                                 + 0.930511 w_k1_IBM_s010
                                 + 1.00082 w_k1_INTC_s010
                                 + 0.989416 w_k1_KO_s010
                                 + 1.08231 w_k1_MAT_s010
                                 + 1.00256 w_k1_MSFT_s010
                                 + 1.07323 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.676731 w_k1_ADM_s010
                                 + 0.792621 w_k1_AXP_s010
                                 + 0.724598 w_k1_DIS_s010
                                 + 0.808498 w_k1_FDX_s010
                                 + 0.799791 w_k1_IBM_s010
                                 + 0.88556 w_k1_INTC_s010
                                 + 0.781132 w_k1_KO_s010
                                 + 0.753563 w_k1_MAT_s010
                                 + 0.868971 w_k1_MSFT_s010
                                 + 0.913694 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.909333 w_k1_ADM_s010
                                 + 0.942577 w_k1_AXP_s010
                                 + 0.594377 w_k1_DIS_s010
                                 + 0.819084 w_k1_FDX_s010
                                 + 0.907781 w_k1_IBM_s010
                                 + 0.821326 w_k1_INTC_s010
                                 + 0.701147 w_k1_KO_s010
                                 + 0.901126 w_k1_MAT_s010
                                 + 0.685204 w_k1_MSFT_s010
                                 + 0.830038 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.29865 w_k1_ADM_s010
                                 + 1.6051 w_k1_AXP_s010 + 1.50643 w_k1_DIS_s010
                                 + 1.31608 w_k1_FDX_s010
                                 + 1.42542 w_k1_IBM_s010
                                 + 1.51524 w_k1_INTC_s010
                                 + 1.21264 w_k1_KO_s010 + 1.35789 w_k1_MAT_s010
                                 + 1.23324 w_k1_MSFT_s010
                                 + 1.09837 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.976333 w_k1_ADM_s010
                                 + 1.05311 w_k1_AXP_s010
                                 + 1.48842 w_k1_DIS_s010
                                 + 1.22735 w_k1_FDX_s010
                                 + 1.03661 w_k1_IBM_s010
                                 + 1.28498 w_k1_INTC_s010 + 1.0126 w_k1_KO_s010
                                 + 0.97611 w_k1_MAT_s010
                                 + 1.11123 w_k1_MSFT_s010
                                 + 1.14507 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.909912 w_k1_ADM_s010
                                 + 1.15692 w_k1_AXP_s010
                                 + 0.90882 w_k1_DIS_s010
                                 + 1.00335 w_k1_FDX_s010
                                 + 0.865868 w_k1_IBM_s010
                                 + 1.03306 w_k1_INTC_s010
                                 + 0.916248 w_k1_KO_s010
                                 + 0.824858 w_k1_MAT_s010
                                 + 0.952637 w_k1_MSFT_s010
                                 + 0.983884 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.00993 w_k1_ADM_s010
                                 + 1.13802 w_k1_AXP_s010
                                 + 1.06191 w_k1_DIS_s010
                                 + 0.941295 w_k1_FDX_s010
                                 + 1.41564 w_k1_IBM_s010
                                 + 1.09411 w_k1_INTC_s010
                                 + 1.14853 w_k1_KO_s010 + 1.37258 w_k1_MAT_s010
                                 + 1.00307 w_k1_MSFT_s010
                                 + 1.08414 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.22086 w_k1_ADM_s011
                                 + 0.99045 w_k1_AXP_s011
                                 + 0.996039 w_k1_DIS_s011
                                 + 1.01619 w_k1_FDX_s011
                                 + 1.15531 w_k1_IBM_s011
                                 + 0.971325 w_k1_INTC_s011
                                 + 1.11186 w_k1_KO_s011 + 1.16387 w_k1_MAT_s011
                                 + 0.96787 w_k1_MSFT_s011
                                 + 1.05092 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.951555 w_k1_ADM_s011
                                 + 0.739492 w_k1_AXP_s011
                                 + 0.736832 w_k1_DIS_s011
                                 + 0.914131 w_k1_FDX_s011
                                 + 0.659884 w_k1_IBM_s011
                                 + 1.10345 w_k1_INTC_s011
                                 + 0.733876 w_k1_KO_s011
                                 + 0.551682 w_k1_MAT_s011
                                 + 0.96512 w_k1_MSFT_s011
                                 + 0.889872 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.16158 w_k1_ADM_s011
                                 + 1.16894 w_k1_AXP_s011
                                 + 1.45731 w_k1_DIS_s011
                                 + 1.14348 w_k1_FDX_s011
                                 + 1.48881 w_k1_IBM_s011
                                 + 1.13125 w_k1_INTC_s011
                                 + 1.44262 w_k1_KO_s011 + 1.1948 w_k1_MAT_s011
                                 + 1.25071 w_k1_MSFT_s011
                                 + 1.19509 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.979443 w_k1_ADM_s011
                                 + 1.08068 w_k1_AXP_s011
                                 + 0.693316 w_k1_DIS_s011
                                 + 0.946445 w_k1_FDX_s011
                                 + 0.831872 w_k1_IBM_s011
                                 + 1.00547 w_k1_INTC_s011
                                 + 0.837316 w_k1_KO_s011
                                 + 0.847233 w_k1_MAT_s011
                                 + 0.892543 w_k1_MSFT_s011
                                 + 0.87154 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.00036 w_k1_ADM_s011
                                 + 0.880003 w_k1_AXP_s011
                                 + 1.00841 w_k1_DIS_s011 + 1.1847 w_k1_FDX_s011
                                 + 0.680031 w_k1_IBM_s011
                                 + 0.965709 w_k1_INTC_s011
                                 + 0.834252 w_k1_KO_s011
                                 + 0.781546 w_k1_MAT_s011
                                 + 1.02809 w_k1_MSFT_s011
                                 + 1.01927 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.797526 w_k1_ADM_s011
                                 + 0.680023 w_k1_AXP_s011
                                 + 0.781326 w_k1_DIS_s011
                                 + 0.810368 w_k1_FDX_s011
                                 + 1.01497 w_k1_IBM_s011
                                 + 0.835064 w_k1_INTC_s011
                                 + 0.912434 w_k1_KO_s011
                                 + 0.842789 w_k1_MAT_s011
                                 + 0.895773 w_k1_MSFT_s011
                                 + 1.00272 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.768588 w_k1_ADM_s011
                                 + 0.686074 w_k1_AXP_s011
                                 + 0.547891 w_k1_DIS_s011
                                 + 0.749355 w_k1_FDX_s011
                                 + 0.751849 w_k1_IBM_s011
                                 + 0.735762 w_k1_INTC_s011
                                 + 0.734039 w_k1_KO_s011
                                 + 0.68529 w_k1_MAT_s011
                                 + 0.755389 w_k1_MSFT_s011
                                 + 0.853044 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.26501 w_k1_ADM_s011
                                 + 1.02611 w_k1_AXP_s011
                                 + 1.24019 w_k1_DIS_s011
                                 + 1.19547 w_k1_FDX_s011
                                 + 0.93844 w_k1_IBM_s011
                                 + 1.04978 w_k1_INTC_s011
                                 + 1.04658 w_k1_KO_s011 + 1.04481 w_k1_MAT_s011
                                 + 1.18644 w_k1_MSFT_s011
                                 + 1.08653 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.03157 w_k1_ADM_s012
                                 + 0.867609 w_k1_AXP_s012
                                 + 0.705229 w_k1_DIS_s012
                                 + 0.836317 w_k1_FDX_s012
                                 + 0.806377 w_k1_IBM_s012
                                 + 0.947763 w_k1_INTC_s012
                                 + 0.742068 w_k1_KO_s012
                                 + 0.513308 w_k1_MAT_s012
                                 + 0.847007 w_k1_MSFT_s012
                                 + 0.861976 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.989708 w_k1_ADM_s012
                                 + 0.987863 w_k1_AXP_s012
                                 + 0.826995 w_k1_DIS_s012
                                 + 0.985793 w_k1_FDX_s012
                                 + 0.690036 w_k1_IBM_s012
                                 + 0.854155 w_k1_INTC_s012
                                 + 0.768712 w_k1_KO_s012
                                 + 0.732209 w_k1_MAT_s012
                                 + 0.91536 w_k1_MSFT_s012
                                 + 0.963051 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.23088 w_k1_ADM_s012
                                 + 0.924808 w_k1_AXP_s012
                                 + 0.933438 w_k1_DIS_s012
                                 + 0.988093 w_k1_FDX_s012
                                 + 1.08139 w_k1_IBM_s012
                                 + 1.04888 w_k1_INTC_s012
                                 + 0.93057 w_k1_KO_s012
                                 + 0.662457 w_k1_MAT_s012
                                 + 0.93353 w_k1_MSFT_s012
                                 + 0.960751 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08497 w_k1_ADM_s012
                                 + 0.989437 w_k1_AXP_s012
                                 + 1.1817 w_k1_DIS_s012
                                 + 0.979697 w_k1_FDX_s012
                                 + 1.54179 w_k1_IBM_s012
                                 + 1.08413 w_k1_INTC_s012
                                 + 1.40862 w_k1_KO_s012 + 1.19155 w_k1_MAT_s012
                                 + 1.05648 w_k1_MSFT_s012
                                 + 1.01757 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.29975 w_k1_ADM_s012
                                 + 1.08636 w_k1_AXP_s012
                                 + 1.22551 w_k1_DIS_s012
                                 + 1.13685 w_k1_FDX_s012
                                 + 0.852091 w_k1_IBM_s012
                                 + 1.21561 w_k1_INTC_s012
                                 + 1.01342 w_k1_KO_s012
                                 + 0.905788 w_k1_MAT_s012
                                 + 1.14883 w_k1_MSFT_s012
                                 + 1.04416 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.864658 w_k1_ADM_s012
                                 + 0.61503 w_k1_AXP_s012
                                 + 0.955057 w_k1_DIS_s012
                                 + 0.841059 w_k1_FDX_s012
                                 + 0.878112 w_k1_IBM_s012
                                 + 0.829711 w_k1_INTC_s012
                                 + 1.04056 w_k1_KO_s012
                                 + 0.700876 w_k1_MAT_s012
                                 + 0.906966 w_k1_MSFT_s012
                                 + 0.928376 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.905219 w_k1_ADM_s012
                                 + 0.899263 w_k1_AXP_s012
                                 + 0.747272 w_k1_DIS_s012
                                 + 0.755081 w_k1_FDX_s012
                                 + 1.11209 w_k1_IBM_s012
                                 + 0.918773 w_k1_INTC_s012
                                 + 0.891345 w_k1_KO_s012
                                 + 0.876729 w_k1_MAT_s012
                                 + 0.794053 w_k1_MSFT_s012
                                 + 0.942567 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.25205 w_k1_ADM_s012
                                 + 1.35441 w_k1_AXP_s012
                                 + 1.37822 w_k1_DIS_s012
                                 + 1.27488 w_k1_FDX_s012
                                 + 1.24116 w_k1_IBM_s012
                                 + 1.0034 w_k1_INTC_s012 + 1.19375 w_k1_KO_s012
                                 + 1.37031 w_k1_MAT_s012
                                 + 1.15535 w_k1_MSFT_s012
                                 + 1.26016 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.885292 w_k1_ADM_s013
                                 + 1.01924 w_k1_AXP_s013
                                 + 0.750029 w_k1_DIS_s013
                                 + 0.930321 w_k1_FDX_s013
                                 + 1.15022 w_k1_IBM_s013
                                 + 1.04107 w_k1_INTC_s013
                                 + 0.961404 w_k1_KO_s013
                                 + 1.08501 w_k1_MAT_s013
                                 + 0.872778 w_k1_MSFT_s013
                                 + 0.916507 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.977206 w_k1_ADM_s013
                                 + 1.42993 w_k1_AXP_s013
                                 + 1.46529 w_k1_DIS_s013
                                 + 1.29897 w_k1_FDX_s013
                                 + 1.00465 w_k1_IBM_s013
                                 + 1.32968 w_k1_INTC_s013 + 1.2256 w_k1_KO_s013
                                 + 1.5216 w_k1_MAT_s013
                                 + 1.29978 w_k1_MSFT_s013
                                 + 1.15818 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.893069 w_k1_ADM_s013
                                 + 1.19927 w_k1_AXP_s013
                                 + 1.12982 w_k1_DIS_s013
                                 + 0.986104 w_k1_FDX_s013
                                 + 1.6711 w_k1_IBM_s013
                                 + 1.18336 w_k1_INTC_s013 + 1.4366 w_k1_KO_s013
                                 + 2.00372 w_k1_MAT_s013
                                 + 1.02585 w_k1_MSFT_s013
                                 + 1.11749 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.93639 w_k1_ADM_s013
                                 + 1.02168 w_k1_AXP_s013
                                 + 0.942013 w_k1_DIS_s013
                                 + 1.19807 w_k1_FDX_s013
                                 + 0.612283 w_k1_IBM_s013
                                 + 1.23792 w_k1_INTC_s013
                                 + 0.831217 w_k1_KO_s013
                                 + 0.503727 w_k1_MAT_s013
                                 + 1.07666 w_k1_MSFT_s013
                                 + 0.906742 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.645032 w_k1_ADM_s013
                                 + 0.664681 w_k1_AXP_s013
                                 + 0.48921 w_k1_DIS_s013
                                 + 0.670442 w_k1_FDX_s013
                                 + 0.759773 w_k1_IBM_s013
                                 + 0.860505 w_k1_INTC_s013
                                 + 0.72802 w_k1_KO_s013 + 0.63627 w_k1_MAT_s013
                                 + 0.703371 w_k1_MSFT_s013
                                 + 0.673997 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.831808 w_k1_ADM_s013
                                 + 0.771667 w_k1_AXP_s013
                                 + 0.675862 w_k1_DIS_s013
                                 + 0.926738 w_k1_FDX_s013
                                 + 0.503294 w_k1_IBM_s013
                                 + 0.877726 w_k1_INTC_s013
                                 + 0.632351 w_k1_KO_s013
                                 + 0.909753 w_k1_MAT_s013
                                 + 0.841129 w_k1_MSFT_s013
                                 + 0.817979 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.965351 w_k1_ADM_s013
                                 + 1.06976 w_k1_AXP_s013
                                 + 1.11846 w_k1_DIS_s013
                                 + 1.07215 w_k1_FDX_s013
                                 + 0.954883 w_k1_IBM_s013
                                 + 1.13402 w_k1_INTC_s013
                                 + 1.25422 w_k1_KO_s013 + 1.18949 w_k1_MAT_s013
                                 + 1.06907 w_k1_MSFT_s013
                                 + 0.997844 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.816555 w_k1_ADM_s013
                                 + 0.802984 w_k1_AXP_s013
                                 + 0.761024 w_k1_DIS_s013
                                 + 0.893996 w_k1_FDX_s013
                                 + 0.810367 w_k1_IBM_s013
                                 + 0.801323 w_k1_INTC_s013
                                 + 0.798991 w_k1_KO_s013
                                 + 0.924802 w_k1_MAT_s013
                                 + 0.889976 w_k1_MSFT_s013
                                 + 0.857502 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.17622 w_k1_ADM_s014
                                 + 1.13274 w_k1_AXP_s014
                                 + 1.24696 w_k1_DIS_s014
                                 + 1.06083 w_k1_FDX_s014 + 1.1019 w_k1_IBM_s014
                                 + 1.01616 w_k1_INTC_s014
                                 + 1.24242 w_k1_KO_s014 + 1.32352 w_k1_MAT_s014
                                 + 1.09066 w_k1_MSFT_s014
                                 + 1.11703 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.843792 w_k1_ADM_s014
                                 + 0.878015 w_k1_AXP_s014
                                 + 0.813688 w_k1_DIS_s014
                                 + 0.803916 w_k1_FDX_s014
                                 + 1.03333 w_k1_IBM_s014
                                 + 0.852059 w_k1_INTC_s014
                                 + 0.955025 w_k1_KO_s014
                                 + 1.15669 w_k1_MAT_s014
                                 + 0.855879 w_k1_MSFT_s014
                                 + 0.948696 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.1291 w_k1_ADM_s014
                                 + 1.40757 w_k1_AXP_s014
                                 + 1.10138 w_k1_DIS_s014
                                 + 1.18542 w_k1_FDX_s014
                                 + 0.832727 w_k1_IBM_s014
                                 + 1.19994 w_k1_INTC_s014
                                 + 0.934825 w_k1_KO_s014
                                 + 0.930261 w_k1_MAT_s014
                                 + 1.01602 w_k1_MSFT_s014 + 1.0073 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.10364 w_k1_ADM_s014
                                 + 1.2645 w_k1_AXP_s014 + 1.61237 w_k1_DIS_s014
                                 + 1.27554 w_k1_FDX_s014
                                 + 1.10612 w_k1_IBM_s014
                                 + 0.949984 w_k1_INTC_s014
                                 + 1.2547 w_k1_KO_s014 + 1.35094 w_k1_MAT_s014
                                 + 1.06268 w_k1_MSFT_s014
                                 + 1.20227 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.740939 w_k1_ADM_s014
                                 + 0.757612 w_k1_AXP_s014
                                 + 0.676157 w_k1_DIS_s014
                                 + 0.799347 w_k1_FDX_s014
                                 + 0.684321 w_k1_IBM_s014
                                 + 0.80451 w_k1_INTC_s014
                                 + 0.73802 w_k1_KO_s014
                                 + 0.671041 w_k1_MAT_s014
                                 + 0.7804 w_k1_MSFT_s014
                                 + 0.881379 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.798176 w_k1_ADM_s014
                                 + 1.02757 w_k1_AXP_s014
                                 + 0.932088 w_k1_DIS_s014
                                 + 0.871465 w_k1_FDX_s014
                                 + 1.32155 w_k1_IBM_s014
                                 + 0.922254 w_k1_INTC_s014
                                 + 1.09696 w_k1_KO_s014 + 1.29997 w_k1_MAT_s014
                                 + 0.792324 w_k1_MSFT_s014
                                 + 0.995546 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.810438 w_k1_ADM_s014
                                 + 0.964098 w_k1_AXP_s014
                                 + 0.917121 w_k1_DIS_s014
                                 + 0.96967 w_k1_FDX_s014
                                 + 0.745965 w_k1_IBM_s014
                                 + 0.993332 w_k1_INTC_s014
                                 + 0.983306 w_k1_KO_s014
                                 + 0.694392 w_k1_MAT_s014
                                 + 0.97041 w_k1_MSFT_s014
                                 + 0.934591 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.13895 w_k1_ADM_s014
                                 + 1.71602 w_k1_AXP_s014
                                 + 1.73064 w_k1_DIS_s014
                                 + 1.25185 w_k1_FDX_s014
                                 + 1.66348 w_k1_IBM_s014
                                 + 1.23307 w_k1_INTC_s014 + 1.5997 w_k1_KO_s014
                                 + 1.48794 w_k1_MAT_s014
                                 + 1.30401 w_k1_MSFT_s014
                                 + 1.41918 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.12609 w_k1_ADM_s015
                                 + 1.08492 w_k1_AXP_s015
                                 + 0.990226 w_k1_DIS_s015
                                 + 1.05561 w_k1_FDX_s015
                                 + 0.94127 w_k1_IBM_s015
                                 + 1.29794 w_k1_INTC_s015
                                 + 0.918592 w_k1_KO_s015
                                 + 0.886632 w_k1_MAT_s015
                                 + 1.17103 w_k1_MSFT_s015
                                 + 1.11962 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.0252 w_k1_ADM_s015
                                 + 0.926956 w_k1_AXP_s015
                                 + 1.01738 w_k1_DIS_s015
                                 + 0.962629 w_k1_FDX_s015
                                 + 1.23148 w_k1_IBM_s015
                                 + 1.15336 w_k1_INTC_s015
                                 + 1.14237 w_k1_KO_s015 + 1.07527 w_k1_MAT_s015
                                 + 1.14231 w_k1_MSFT_s015
                                 + 1.02269 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.779709 w_k1_ADM_s015
                                 + 0.88656 w_k1_AXP_s015
                                 + 0.650551 w_k1_DIS_s015
                                 + 0.872868 w_k1_FDX_s015
                                 + 0.693832 w_k1_IBM_s015
                                 + 0.901617 w_k1_INTC_s015
                                 + 0.614091 w_k1_KO_s015
                                 + 0.675111 w_k1_MAT_s015
                                 + 0.958073 w_k1_MSFT_s015
                                 + 0.936313 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.05985 w_k1_ADM_s015
                                 + 1.13672 w_k1_AXP_s015
                                 + 1.05967 w_k1_DIS_s015
                                 + 0.898328 w_k1_FDX_s015
                                 + 1.8658 w_k1_IBM_s015
                                 + 1.28475 w_k1_INTC_s015
                                 + 1.29012 w_k1_KO_s015 + 1.33728 w_k1_MAT_s015
                                 + 1.05524 w_k1_MSFT_s015
                                 + 1.18882 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.22988 w_k1_ADM_s015
                                 + 1.46998 w_k1_AXP_s015
                                 + 1.50694 w_k1_DIS_s015
                                 + 1.33617 w_k1_FDX_s015 + 1.2942 w_k1_IBM_s015
                                 + 1.41673 w_k1_INTC_s015
                                 + 1.28693 w_k1_KO_s015 + 1.52229 w_k1_MAT_s015
                                 + 1.39177 w_k1_MSFT_s015
                                 + 1.40211 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.82774 w_k1_ADM_s015
                                 + 0.71585 w_k1_AXP_s015
                                 + 0.599276 w_k1_DIS_s015
                                 + 0.739062 w_k1_FDX_s015
                                 + 0.941605 w_k1_IBM_s015
                                 + 0.820009 w_k1_INTC_s015
                                 + 0.814663 w_k1_KO_s015
                                 + 0.798286 w_k1_MAT_s015
                                 + 0.878636 w_k1_MSFT_s015
                                 + 0.909085 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.790483 w_k1_ADM_s015
                                 + 0.746925 w_k1_AXP_s015
                                 + 0.835824 w_k1_DIS_s015
                                 + 0.815885 w_k1_FDX_s015
                                 + 1.00193 w_k1_IBM_s015
                                 + 1.01226 w_k1_INTC_s015
                                 + 0.885883 w_k1_KO_s015
                                 + 0.843629 w_k1_MAT_s015
                                 + 0.929829 w_k1_MSFT_s015
                                 + 1.01993 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.993583 w_k1_ADM_s015
                                 + 1.05532 w_k1_AXP_s015
                                 + 1.10414 w_k1_DIS_s015
                                 + 1.12192 w_k1_FDX_s015
                                 + 0.658111 w_k1_IBM_s015
                                 + 1.36477 w_k1_INTC_s015
                                 + 0.967731 w_k1_KO_s015
                                 + 0.656379 w_k1_MAT_s015
                                 + 1.22518 w_k1_MSFT_s015
                                 + 0.989673 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.13408 w_k1_ADM_s016
                                 + 1.22144 w_k1_AXP_s016
                                 + 1.32671 w_k1_DIS_s016
                                 + 1.18652 w_k1_FDX_s016
                                 + 0.920535 w_k1_IBM_s016
                                 + 1.22523 w_k1_INTC_s016
                                 + 1.07262 w_k1_KO_s016
                                 + 0.858201 w_k1_MAT_s016
                                 + 1.17639 w_k1_MSFT_s016 + 1.0105 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.11692 w_k1_ADM_s016
                                 + 0.698412 w_k1_AXP_s016
                                 + 0.798396 w_k1_DIS_s016
                                 + 0.826722 w_k1_FDX_s016
                                 + 0.666722 w_k1_IBM_s016
                                 + 0.875907 w_k1_INTC_s016
                                 + 0.897439 w_k1_KO_s016
                                 + 0.487956 w_k1_MAT_s016
                                 + 0.837561 w_k1_MSFT_s016
                                 + 0.843174 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.22854 w_k1_ADM_s016
                                 + 1.19096 w_k1_AXP_s016
                                 + 1.45616 w_k1_DIS_s016
                                 + 1.16396 w_k1_FDX_s016
                                 + 1.85373 w_k1_IBM_s016
                                 + 1.06726 w_k1_INTC_s016
                                 + 1.40812 w_k1_KO_s016 + 1.54983 w_k1_MAT_s016
                                 + 1.16302 w_k1_MSFT_s016
                                 + 1.05653 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.902857 w_k1_ADM_s016
                                 + 0.833247 w_k1_AXP_s016
                                 + 0.699089 w_k1_DIS_s016
                                 + 0.787528 w_k1_FDX_s016
                                 + 0.859258 w_k1_IBM_s016
                                 + 0.962498 w_k1_INTC_s016
                                 + 0.820997 w_k1_KO_s016
                                 + 0.622077 w_k1_MAT_s016
                                 + 0.852711 w_k1_MSFT_s016
                                 + 0.840672 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.21081 w_k1_ADM_s016
                                 + 1.20218 w_k1_AXP_s016
                                 + 0.96432 w_k1_DIS_s016
                                 + 1.08628 w_k1_FDX_s016
                                 + 0.726669 w_k1_IBM_s016
                                 + 1.12048 w_k1_INTC_s016
                                 + 0.918908 w_k1_KO_s016
                                 + 0.586221 w_k1_MAT_s016
                                 + 0.992624 w_k1_MSFT_s016
                                 + 0.997169 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.870773 w_k1_ADM_s016
                                 + 0.866482 w_k1_AXP_s016
                                 + 0.890834 w_k1_DIS_s016
                                 + 0.862874 w_k1_FDX_s016
                                 + 1.25035 w_k1_IBM_s016
                                 + 1.00443 w_k1_INTC_s016
                                 + 1.08123 w_k1_KO_s016
                                 + 0.727069 w_k1_MAT_s016
                                 + 0.923794 w_k1_MSFT_s016
                                 + 0.932153 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.32531 w_k1_ADM_s016
                                 + 1.25474 w_k1_AXP_s016
                                 + 1.12739 w_k1_DIS_s016
                                 + 1.05648 w_k1_FDX_s016
                                 + 1.22653 w_k1_IBM_s016
                                 + 1.11405 w_k1_INTC_s016
                                 + 1.25175 w_k1_KO_s016 + 1.28559 w_k1_MAT_s016
                                 + 1.02701 w_k1_MSFT_s016
                                 + 1.07924 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.46961 w_k1_ADM_s016
                                 + 1.41445 w_k1_AXP_s016
                                 + 1.59306 w_k1_DIS_s016
                                 + 1.37343 w_k1_FDX_s016
                                 + 1.23022 w_k1_IBM_s016
                                 + 1.26713 w_k1_INTC_s016
                                 + 1.34262 w_k1_KO_s016 + 1.29104 w_k1_MAT_s016
                                 + 1.38054 w_k1_MSFT_s016
                                 + 1.25713 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.05684 w_k1_ADM_s017
                                 + 0.902895 w_k1_AXP_s017
                                 + 1.25636 w_k1_DIS_s017
                                 + 1.00749 w_k1_FDX_s017
                                 + 0.970242 w_k1_IBM_s017
                                 + 1.10172 w_k1_INTC_s017
                                 + 1.20715 w_k1_KO_s017
                                 + 0.767331 w_k1_MAT_s017
                                 + 1.1398 w_k1_MSFT_s017 + 1.1067 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.33313 w_k1_ADM_s017
                                 + 1.41242 w_k1_AXP_s017
                                 + 0.992502 w_k1_DIS_s017
                                 + 1.18046 w_k1_FDX_s017
                                 + 0.883947 w_k1_IBM_s017
                                 + 1.03627 w_k1_INTC_s017
                                 + 0.986641 w_k1_KO_s017
                                 + 1.10293 w_k1_MAT_s017
                                 + 1.03687 w_k1_MSFT_s017
                                 + 1.17964 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.954405 w_k1_ADM_s017
                                 + 1.02994 w_k1_AXP_s017
                                 + 0.940558 w_k1_DIS_s017
                                 + 0.949731 w_k1_FDX_s017
                                 + 1.00949 w_k1_IBM_s017
                                 + 0.790093 w_k1_INTC_s017
                                 + 0.940099 w_k1_KO_s017
                                 + 0.858035 w_k1_MAT_s017
                                 + 1.00272 w_k1_MSFT_s017
                                 + 1.03562 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.11458 w_k1_ADM_s017
                                 + 0.731636 w_k1_AXP_s017
                                 + 0.840922 w_k1_DIS_s017
                                 + 1.02714 w_k1_FDX_s017
                                 + 0.654949 w_k1_IBM_s017
                                 + 0.768041 w_k1_INTC_s017
                                 + 0.764324 w_k1_KO_s017
                                 + 0.570325 w_k1_MAT_s017
                                 + 0.869403 w_k1_MSFT_s017
                                 + 0.939744 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.831851 w_k1_ADM_s017
                                 + 0.757397 w_k1_AXP_s017
                                 + 0.655202 w_k1_DIS_s017
                                 + 0.772761 w_k1_FDX_s017
                                 + 0.777361 w_k1_IBM_s017
                                 + 0.702649 w_k1_INTC_s017
                                 + 0.809412 w_k1_KO_s017
                                 + 0.994707 w_k1_MAT_s017
                                 + 0.83052 w_k1_MSFT_s017
                                 + 0.923713 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.33252 w_k1_ADM_s017
                                 + 1.19943 w_k1_AXP_s017
                                 + 1.31535 w_k1_DIS_s017
                                 + 1.18048 w_k1_FDX_s017
                                 + 0.958228 w_k1_IBM_s017
                                 + 0.810558 w_k1_INTC_s017
                                 + 1.12362 w_k1_KO_s017 + 1.01881 w_k1_MAT_s017
                                 + 1.10334 w_k1_MSFT_s017
                                 + 1.14454 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.35509 w_k1_ADM_s017
                                 + 1.48966 w_k1_AXP_s017
                                 + 1.57683 w_k1_DIS_s017
                                 + 1.45564 w_k1_FDX_s017
                                 + 1.27229 w_k1_IBM_s017
                                 + 0.980101 w_k1_INTC_s017
                                 + 1.27252 w_k1_KO_s017 + 1.32535 w_k1_MAT_s017
                                 + 1.27517 w_k1_MSFT_s017
                                 + 1.17571 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.88136 w_k1_ADM_s017
                                 + 1.0292 w_k1_AXP_s017 + 1.05439 w_k1_DIS_s017
                                 + 0.937702 w_k1_FDX_s017
                                 + 1.56725 w_k1_IBM_s017
                                 + 0.759274 w_k1_INTC_s017
                                 + 1.24535 w_k1_KO_s017 + 1.74479 w_k1_MAT_s017
                                 + 0.975733 w_k1_MSFT_s017
                                 + 1.03754 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.780091 w_k1_ADM_s100
                                 + 0.701258 w_k1_AXP_s100
                                 + 0.605649 w_k1_DIS_s100
                                 + 0.744219 w_k1_FDX_s100
                                 + 0.682594 w_k1_IBM_s100
                                 + 0.723704 w_k1_INTC_s100
                                 + 0.63029 w_k1_KO_s100
                                 + 0.689763 w_k1_MAT_s100
                                 + 0.714807 w_k1_MSFT_s100
                                 + 0.837488 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.00556 w_k1_ADM_s100
                                 + 1.32313 w_k1_AXP_s100
                                 + 1.36517 w_k1_DIS_s100 + 1.1998 w_k1_FDX_s100
                                 + 1.32529 w_k1_IBM_s100
                                 + 1.20616 w_k1_INTC_s100
                                 + 1.03708 w_k1_KO_s100 + 1.47341 w_k1_MAT_s100
                                 + 1.06994 w_k1_MSFT_s100
                                 + 1.20597 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.76696 w_k1_ADM_s100
                                 + 0.814566 w_k1_AXP_s100
                                 + 0.882584 w_k1_DIS_s100
                                 + 0.979099 w_k1_FDX_s100
                                 + 0.731332 w_k1_IBM_s100
                                 + 1.00457 w_k1_INTC_s100 + 0.7881 w_k1_KO_s100
                                 + 0.701465 w_k1_MAT_s100
                                 + 0.955193 w_k1_MSFT_s100
                                 + 0.887495 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.02686 w_k1_ADM_s100
                                 + 1.13842 w_k1_AXP_s100 + 1.119 w_k1_DIS_s100
                                 + 1.12724 w_k1_FDX_s100
                                 + 0.845948 w_k1_IBM_s100
                                 + 1.21214 w_k1_INTC_s100
                                 + 1.02512 w_k1_KO_s100
                                 + 0.828769 w_k1_MAT_s100
                                 + 1.01985 w_k1_MSFT_s100
                                 + 1.17943 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.937533 w_k1_ADM_s100
                                 + 1.36263 w_k1_AXP_s100
                                 + 1.01162 w_k1_DIS_s100
                                 + 1.04304 w_k1_FDX_s100
                                 + 1.08042 w_k1_IBM_s100
                                 + 1.15198 w_k1_INTC_s100
                                 + 0.949865 w_k1_KO_s100 + 1.2363 w_k1_MAT_s100
                                 + 0.97742 w_k1_MSFT_s100
                                 + 1.09105 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.855317 w_k1_ADM_s100
                                 + 1.01298 w_k1_AXP_s100
                                 + 1.20323 w_k1_DIS_s100
                                 + 0.982507 w_k1_FDX_s100
                                 + 1.09676 w_k1_IBM_s100
                                 + 1.09291 w_k1_INTC_s100
                                 + 0.960603 w_k1_KO_s100
                                 + 1.00114 w_k1_MAT_s100
                                 + 0.988644 w_k1_MSFT_s100
                                 + 0.988913 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.27755 w_k1_ADM_s100
                                 + 1.46819 w_k1_AXP_s100
                                 + 1.43246 w_k1_DIS_s100
                                 + 1.22292 w_k1_FDX_s100
                                 + 1.90867 w_k1_IBM_s100
                                 + 1.1904 w_k1_INTC_s100 + 1.4374 w_k1_KO_s100
                                 + 1.83132 w_k1_MAT_s100
                                 + 1.14075 w_k1_MSFT_s100
                                 + 1.41878 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.666444 w_k1_ADM_s100
                                 + 0.725162 w_k1_AXP_s100
                                 + 0.732386 w_k1_DIS_s100
                                 + 0.735028 w_k1_FDX_s100
                                 + 1.0127 w_k1_IBM_s100
                                 + 0.965974 w_k1_INTC_s100
                                 + 0.972012 w_k1_KO_s100
                                 + 0.720059 w_k1_MAT_s100
                                 + 0.765287 w_k1_MSFT_s100
                                 + 0.915512 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.769235 w_k1_ADM_s101
                                 + 0.811696 w_k1_AXP_s101
                                 + 0.697726 w_k1_DIS_s101
                                 + 0.925574 w_k1_FDX_s101
                                 + 0.78518 w_k1_IBM_s101
                                 + 0.904033 w_k1_INTC_s101
                                 + 0.776593 w_k1_KO_s101
                                 + 0.808569 w_k1_MAT_s101
                                 + 0.930209 w_k1_MSFT_s101
                                 + 0.874963 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.862075 w_k1_ADM_s101
                                 + 1.72169 w_k1_AXP_s101
                                 + 1.32159 w_k1_DIS_s101
                                 + 1.13817 w_k1_FDX_s101
                                 + 1.78529 w_k1_IBM_s101
                                 + 1.23132 w_k1_INTC_s101
                                 + 1.43847 w_k1_KO_s101 + 2.14615 w_k1_MAT_s101
                                 + 1.17722 w_k1_MSFT_s101
                                 + 1.12984 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.887185 w_k1_ADM_s101
                                 + 1.59201 w_k1_AXP_s101
                                 + 1.62786 w_k1_DIS_s101
                                 + 1.30993 w_k1_FDX_s101
                                 + 1.19072 w_k1_IBM_s101
                                 + 1.35028 w_k1_INTC_s101
                                 + 1.24393 w_k1_KO_s101 + 1.23195 w_k1_MAT_s101
                                 + 1.32194 w_k1_MSFT_s101
                                 + 1.15624 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.792391 w_k1_ADM_s101
                                 + 1.24671 w_k1_AXP_s101
                                 + 0.956025 w_k1_DIS_s101
                                 + 0.978644 w_k1_FDX_s101
                                 + 1.31688 w_k1_IBM_s101
                                 + 1.09772 w_k1_INTC_s101
                                 + 1.23866 w_k1_KO_s101 + 1.81455 w_k1_MAT_s101
                                 + 1.08027 w_k1_MSFT_s101
                                 + 1.25274 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.82732 w_k1_ADM_s101
                                 + 0.676933 w_k1_AXP_s101
                                 + 0.676602 w_k1_DIS_s101
                                 + 0.877163 w_k1_FDX_s101
                                 + 0.580503 w_k1_IBM_s101
                                 + 0.948373 w_k1_INTC_s101
                                 + 0.698203 w_k1_KO_s101
                                 + 0.533821 w_k1_MAT_s101
                                 + 0.877188 w_k1_MSFT_s101
                                 + 0.851299 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.88314 w_k1_ADM_s101
                                 + 1.07793 w_k1_AXP_s101
                                 + 0.864087 w_k1_DIS_s101
                                 + 1.0131 w_k1_FDX_s101 + 0.97454 w_k1_IBM_s101
                                 + 1.09009 w_k1_INTC_s101
                                 + 0.917889 w_k1_KO_s101
                                 + 1.08097 w_k1_MAT_s101
                                 + 1.05364 w_k1_MSFT_s101
                                 + 0.991326 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.943529 w_k1_ADM_s101
                                 + 0.954914 w_k1_AXP_s101
                                 + 1.07342 w_k1_DIS_s101
                                 + 1.01524 w_k1_FDX_s101
                                 + 1.21332 w_k1_IBM_s101
                                 + 1.03048 w_k1_INTC_s101
                                 + 1.04182 w_k1_KO_s101 + 1.09689 w_k1_MAT_s101
                                 + 1.02868 w_k1_MSFT_s101
                                 + 0.873378 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.06714 w_k1_ADM_s101
                                 + 1.24718 w_k1_AXP_s101
                                 + 1.28613 w_k1_DIS_s101
                                 + 1.38357 w_k1_FDX_s101
                                 + 0.828937 w_k1_IBM_s101
                                 + 1.15444 w_k1_INTC_s101
                                 + 0.927051 w_k1_KO_s101
                                 + 0.918975 w_k1_MAT_s101
                                 + 1.16972 w_k1_MSFT_s101
                                 + 1.03978 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.867729 w_k1_ADM_s102
                                 + 1.01781 w_k1_AXP_s102
                                 + 0.748937 w_k1_DIS_s102
                                 + 0.843151 w_k1_FDX_s102
                                 + 1.01056 w_k1_IBM_s102
                                 + 1.13374 w_k1_INTC_s102
                                 + 0.865714 w_k1_KO_s102
                                 + 0.693887 w_k1_MAT_s102
                                 + 0.880124 w_k1_MSFT_s102
                                 + 0.95557 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.06065 w_k1_ADM_s102
                                 + 1.32553 w_k1_AXP_s102 + 1.5321 w_k1_DIS_s102
                                 + 1.34684 w_k1_FDX_s102
                                 + 0.945637 w_k1_IBM_s102
                                 + 1.18663 w_k1_INTC_s102
                                 + 1.05411 w_k1_KO_s102
                                 + 0.672569 w_k1_MAT_s102
                                 + 1.18504 w_k1_MSFT_s102 + 1.2138 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.988533 w_k1_ADM_s102
                                 + 1.12061 w_k1_AXP_s102
                                 + 0.927995 w_k1_DIS_s102
                                 + 1.01224 w_k1_FDX_s102
                                 + 0.873241 w_k1_IBM_s102
                                 + 0.902916 w_k1_INTC_s102
                                 + 0.827628 w_k1_KO_s102
                                 + 0.718127 w_k1_MAT_s102
                                 + 0.847121 w_k1_MSFT_s102
                                 + 0.988892 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.817228 w_k1_ADM_s102
                                 + 1.43008 w_k1_AXP_s102
                                 + 1.11612 w_k1_DIS_s102
                                 + 1.06474 w_k1_FDX_s102 + 1.4753 w_k1_IBM_s102
                                 + 1.11474 w_k1_INTC_s102
                                 + 1.18321 w_k1_KO_s102 + 1.43278 w_k1_MAT_s102
                                 + 1.02586 w_k1_MSFT_s102
                                 + 1.04549 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.694364 w_k1_ADM_s102
                                 + 0.807371 w_k1_AXP_s102
                                 + 0.643885 w_k1_DIS_s102
                                 + 0.736409 w_k1_FDX_s102
                                 + 1.05481 w_k1_IBM_s102
                                 + 0.809756 w_k1_INTC_s102
                                 + 0.811494 w_k1_KO_s102
                                 + 0.722055 w_k1_MAT_s102
                                 + 0.768883 w_k1_MSFT_s102
                                 + 0.876437 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.928583 w_k1_ADM_s102
                                 + 1.05543 w_k1_AXP_s102
                                 + 1.25703 w_k1_DIS_s102
                                 + 1.14756 w_k1_FDX_s102
                                 + 1.23792 w_k1_IBM_s102
                                 + 1.17675 w_k1_INTC_s102
                                 + 1.07891 w_k1_KO_s102 + 1.19277 w_k1_MAT_s102
                                 + 1.11138 w_k1_MSFT_s102
                                 + 1.08825 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.945381 w_k1_ADM_s102
                                 + 0.915568 w_k1_AXP_s102
                                 + 0.983398 w_k1_DIS_s102
                                 + 1.13147 w_k1_FDX_s102
                                 + 0.672137 w_k1_IBM_s102
                                 + 0.946309 w_k1_INTC_s102
                                 + 0.704017 w_k1_KO_s102
                                 + 0.620277 w_k1_MAT_s102
                                 + 1.11133 w_k1_MSFT_s102
                                 + 0.945541 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.24412 w_k1_ADM_s102
                                 + 1.34549 w_k1_AXP_s102
                                 + 1.33067 w_k1_DIS_s102
                                 + 1.04331 w_k1_FDX_s102
                                 + 1.90961 w_k1_IBM_s102
                                 + 0.949149 w_k1_INTC_s102
                                 + 1.15672 w_k1_KO_s102 + 1.45567 w_k1_MAT_s102
                                 + 1.03144 w_k1_MSFT_s102
                                 + 1.16592 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.05156 w_k1_ADM_s103
                                 + 1.20893 w_k1_AXP_s103
                                 + 1.36337 w_k1_DIS_s103 + 1.0379 w_k1_FDX_s103
                                 + 1.4962 w_k1_IBM_s103 + 1.2189 w_k1_INTC_s103
                                 + 1.23392 w_k1_KO_s103 + 1.58373 w_k1_MAT_s103
                                 + 1.19363 w_k1_MSFT_s103
                                 + 1.17013 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.762327 w_k1_ADM_s103
                                 + 0.657279 w_k1_AXP_s103
                                 + 0.923412 w_k1_DIS_s103
                                 + 0.823586 w_k1_FDX_s103
                                 + 0.736067 w_k1_IBM_s103
                                 + 0.91875 w_k1_INTC_s103
                                 + 0.852636 w_k1_KO_s103
                                 + 0.696875 w_k1_MAT_s103
                                 + 1.07772 w_k1_MSFT_s103
                                 + 1.03496 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.872687 w_k1_ADM_s103
                                 + 0.927796 w_k1_AXP_s103
                                 + 1.04158 w_k1_DIS_s103
                                 + 0.92908 w_k1_FDX_s103 + 1.1723 w_k1_IBM_s103
                                 + 1.03219 w_k1_INTC_s103
                                 + 0.930117 w_k1_KO_s103 + 1.0503 w_k1_MAT_s103
                                 + 1.09918 w_k1_MSFT_s103 + 0.9683 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.836591 w_k1_ADM_s103
                                 + 0.713436 w_k1_AXP_s103
                                 + 0.695692 w_k1_DIS_s103
                                 + 0.785334 w_k1_FDX_s103
                                 + 0.857287 w_k1_IBM_s103
                                 + 0.825662 w_k1_INTC_s103
                                 + 0.731322 w_k1_KO_s103
                                 + 0.765363 w_k1_MAT_s103
                                 + 0.811942 w_k1_MSFT_s103
                                 + 0.838866 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.30478 w_k1_ADM_s103
                                 + 0.884739 w_k1_AXP_s103
                                 + 1.17769 w_k1_DIS_s103
                                 + 1.19417 w_k1_FDX_s103
                                 + 0.669393 w_k1_IBM_s103
                                 + 1.22307 w_k1_INTC_s103
                                 + 0.894887 w_k1_KO_s103
                                 + 0.547486 w_k1_MAT_s103
                                 + 1.25825 w_k1_MSFT_s103
                                 + 1.03341 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.08381 w_k1_ADM_s103
                                 + 1.03014 w_k1_AXP_s103
                                 + 1.62649 w_k1_DIS_s103
                                 + 1.33514 w_k1_FDX_s103
                                 + 1.05017 w_k1_IBM_s103
                                 + 1.17905 w_k1_INTC_s103
                                 + 1.03021 w_k1_KO_s103 + 1.00269 w_k1_MAT_s103
                                 + 1.42308 w_k1_MSFT_s103 + 1.1021 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.926045 w_k1_ADM_s103
                                 + 1.09759 w_k1_AXP_s103
                                 + 0.999118 w_k1_DIS_s103
                                 + 1.15537 w_k1_FDX_s103
                                 + 0.793602 w_k1_IBM_s103
                                 + 0.865439 w_k1_INTC_s103
                                 + 0.731456 w_k1_KO_s103
                                 + 0.84737 w_k1_MAT_s103
                                 + 1.01796 w_k1_MSFT_s103
                                 + 0.982412 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01309 w_k1_ADM_s103
                                 + 0.952538 w_k1_AXP_s103
                                 + 1.0341 w_k1_DIS_s103
                                 + 0.941379 w_k1_FDX_s103
                                 + 0.875817 w_k1_IBM_s103
                                 + 1.0287 w_k1_INTC_s103
                                 + 0.913591 w_k1_KO_s103
                                 + 0.769543 w_k1_MAT_s103
                                 + 1.01924 w_k1_MSFT_s103
                                 + 0.99388 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.746353 w_k1_ADM_s104
                                 + 0.817802 w_k1_AXP_s104
                                 + 0.802813 w_k1_DIS_s104
                                 + 0.754944 w_k1_FDX_s104
                                 + 1.11681 w_k1_IBM_s104
                                 + 0.863449 w_k1_INTC_s104
                                 + 0.944381 w_k1_KO_s104
                                 + 1.15196 w_k1_MAT_s104
                                 + 0.88689 w_k1_MSFT_s104
                                 + 0.811297 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.810798 w_k1_ADM_s104
                                 + 0.645853 w_k1_AXP_s104
                                 + 0.604119 w_k1_DIS_s104
                                 + 0.765149 w_k1_FDX_s104
                                 + 0.611083 w_k1_IBM_s104
                                 + 0.88347 w_k1_INTC_s104
                                 + 0.645186 w_k1_KO_s104
                                 + 0.787253 w_k1_MAT_s104
                                 + 0.759855 w_k1_MSFT_s104
                                 + 0.711099 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.920957 w_k1_ADM_s104
                                 + 1.07684 w_k1_AXP_s104
                                 + 1.14255 w_k1_DIS_s104
                                 + 0.983707 w_k1_FDX_s104
                                 + 1.1835 w_k1_IBM_s104
                                 + 1.12545 w_k1_INTC_s104
                                 + 1.01565 w_k1_KO_s104 + 1.37265 w_k1_MAT_s104
                                 + 1.1922 w_k1_MSFT_s104
                                 + 0.935201 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.816186 w_k1_ADM_s104
                                 + 1.01281 w_k1_AXP_s104
                                 + 0.93684 w_k1_DIS_s104
                                 + 1.06016 w_k1_FDX_s104
                                 + 0.695116 w_k1_IBM_s104
                                 + 1.11147 w_k1_INTC_s104
                                 + 0.846965 w_k1_KO_s104
                                 + 0.668117 w_k1_MAT_s104
                                 + 1.06125 w_k1_MSFT_s104
                                 + 0.93309 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.01098 w_k1_ADM_s104
                                 + 1.26076 w_k1_AXP_s104
                                 + 1.16298 w_k1_DIS_s104 + 1.2058 w_k1_FDX_s104
                                 + 0.842768 w_k1_IBM_s104
                                 + 1.08544 w_k1_INTC_s104
                                 + 1.03955 w_k1_KO_s104 + 1.29499 w_k1_MAT_s104
                                 + 1.22891 w_k1_MSFT_s104
                                 + 1.07247 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.24767 w_k1_ADM_s104
                                 + 1.34323 w_k1_AXP_s104
                                 + 1.53099 w_k1_DIS_s104
                                 + 1.39385 w_k1_FDX_s104
                                 + 1.06589 w_k1_IBM_s104
                                 + 1.45721 w_k1_INTC_s104
                                 + 1.23113 w_k1_KO_s104 + 1.08707 w_k1_MAT_s104
                                 + 1.47316 w_k1_MSFT_s104 + 1.1178 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.08406 w_k1_ADM_s104
                                 + 1.06669 w_k1_AXP_s104
                                 + 1.39471 w_k1_DIS_s104
                                 + 1.11386 w_k1_FDX_s104
                                 + 1.37447 w_k1_IBM_s104
                                 + 0.977848 w_k1_INTC_s104
                                 + 1.18595 w_k1_KO_s104 + 1.68676 w_k1_MAT_s104
                                 + 1.25913 w_k1_MSFT_s104
                                 + 1.09422 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.937128 w_k1_ADM_s104
                                 + 0.864135 w_k1_AXP_s104
                                 + 0.993428 w_k1_DIS_s104
                                 + 0.951498 w_k1_FDX_s104
                                 + 0.857938 w_k1_IBM_s104
                                 + 0.887825 w_k1_INTC_s104
                                 + 0.870712 w_k1_KO_s104
                                 + 1.05422 w_k1_MAT_s104
                                 + 0.979171 w_k1_MSFT_s104
                                 + 0.909115 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.35957 w_k1_ADM_s105
                                 + 1.25877 w_k1_AXP_s105
                                 + 1.27559 w_k1_DIS_s105
                                 + 1.23362 w_k1_FDX_s105
                                 + 0.975092 w_k1_IBM_s105
                                 + 1.10353 w_k1_INTC_s105
                                 + 1.12959 w_k1_KO_s105 + 1.29721 w_k1_MAT_s105
                                 + 1.24224 w_k1_MSFT_s105
                                 + 1.10641 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.91929 w_k1_ADM_s105
                                 + 0.792337 w_k1_AXP_s105
                                 + 0.782397 w_k1_DIS_s105
                                 + 0.780776 w_k1_FDX_s105
                                 + 1.07555 w_k1_IBM_s105
                                 + 0.856352 w_k1_INTC_s105
                                 + 1.02988 w_k1_KO_s105
                                 + 0.930499 w_k1_MAT_s105
                                 + 1.04257 w_k1_MSFT_s105
                                 + 0.935751 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.56903 w_k1_ADM_s105
                                 + 1.71127 w_k1_AXP_s105
                                 + 1.59968 w_k1_DIS_s105
                                 + 1.31625 w_k1_FDX_s105
                                 + 1.43237 w_k1_IBM_s105
                                 + 1.25033 w_k1_INTC_s105
                                 + 1.36696 w_k1_KO_s105 + 1.43395 w_k1_MAT_s105
                                 + 1.33243 w_k1_MSFT_s105
                                 + 1.21676 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.969341 w_k1_ADM_s105
                                 + 1.30195 w_k1_AXP_s105
                                 + 1.04795 w_k1_DIS_s105
                                 + 1.01405 w_k1_FDX_s105
                                 + 1.41619 w_k1_IBM_s105
                                 + 1.27173 w_k1_INTC_s105
                                 + 1.22731 w_k1_KO_s105 + 1.47559 w_k1_MAT_s105
                                 + 1.16315 w_k1_MSFT_s105
                                 + 1.17585 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.08369 w_k1_ADM_s105
                                 + 1.11325 w_k1_AXP_s105
                                 + 0.990586 w_k1_DIS_s105
                                 + 1.06169 w_k1_FDX_s105
                                 + 1.23318 w_k1_IBM_s105 + 1.271 w_k1_INTC_s105
                                 + 1.0639 w_k1_KO_s105 + 1.21231 w_k1_MAT_s105
                                 + 1.20319 w_k1_MSFT_s105
                                 + 1.12796 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.903446 w_k1_ADM_s105
                                 + 0.968983 w_k1_AXP_s105
                                 + 0.905119 w_k1_DIS_s105
                                 + 0.962031 w_k1_FDX_s105
                                 + 0.896333 w_k1_IBM_s105
                                 + 1.08715 w_k1_INTC_s105
                                 + 0.959584 w_k1_KO_s105
                                 + 0.816828 w_k1_MAT_s105
                                 + 0.988912 w_k1_MSFT_s105
                                 + 0.869308 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.0579 w_k1_ADM_s105
                                 + 1.40677 w_k1_AXP_s105
                                 + 0.866274 w_k1_DIS_s105
                                 + 1.10168 w_k1_FDX_s105
                                 + 0.811078 w_k1_IBM_s105
                                 + 1.08279 w_k1_INTC_s105
                                 + 0.894713 w_k1_KO_s105 + 1.0939 w_k1_MAT_s105
                                 + 1.03482 w_k1_MSFT_s105
                                 + 0.950201 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.860556 w_k1_ADM_s105
                                 + 0.744594 w_k1_AXP_s105
                                 + 0.645755 w_k1_DIS_s105
                                 + 0.82248 w_k1_FDX_s105
                                 + 0.684403 w_k1_IBM_s105
                                 + 0.941192 w_k1_INTC_s105
                                 + 0.729885 w_k1_KO_s105
                                 + 0.602489 w_k1_MAT_s105
                                 + 0.946961 w_k1_MSFT_s105
                                 + 0.824115 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.836368 w_k1_ADM_s106
                                 + 0.768667 w_k1_AXP_s106
                                 + 0.905832 w_k1_DIS_s106
                                 + 0.864475 w_k1_FDX_s106
                                 + 0.958915 w_k1_IBM_s106
                                 + 0.789467 w_k1_INTC_s106
                                 + 0.914818 w_k1_KO_s106
                                 + 0.759127 w_k1_MAT_s106
                                 + 0.862308 w_k1_MSFT_s106
                                 + 0.85806 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.05022 w_k1_ADM_s106
                                 + 1.26023 w_k1_AXP_s106
                                 + 0.860886 w_k1_DIS_s106
                                 + 1.04809 w_k1_FDX_s106
                                 + 1.05056 w_k1_IBM_s106
                                 + 0.892052 w_k1_INTC_s106
                                 + 1.01641 w_k1_KO_s106 + 1.64602 w_k1_MAT_s106
                                 + 1.06171 w_k1_MSFT_s106
                                 + 0.972373 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.04335 w_k1_ADM_s106
                                 + 0.80472 w_k1_AXP_s106 + 1.0321 w_k1_DIS_s106
                                 + 0.924312 w_k1_FDX_s106
                                 + 1.23158 w_k1_IBM_s106
                                 + 1.21281 w_k1_INTC_s106
                                 + 1.09526 w_k1_KO_s106 + 1.393 w_k1_MAT_s106
                                 + 1.20258 w_k1_MSFT_s106
                                 + 0.94934 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.25078 w_k1_ADM_s106
                                 + 1.11104 w_k1_AXP_s106
                                 + 1.47991 w_k1_DIS_s106
                                 + 1.29189 w_k1_FDX_s106
                                 + 0.82046 w_k1_IBM_s106
                                 + 1.02612 w_k1_INTC_s106
                                 + 1.03676 w_k1_KO_s106 + 1.09625 w_k1_MAT_s106
                                 + 1.26896 w_k1_MSFT_s106
                                 + 1.03332 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.03749 w_k1_ADM_s106
                                 + 0.968679 w_k1_AXP_s106
                                 + 0.889987 w_k1_DIS_s106
                                 + 1.04082 w_k1_FDX_s106
                                 + 0.79269 w_k1_IBM_s106
                                 + 0.99304 w_k1_INTC_s106
                                 + 1.02829 w_k1_KO_s106
                                 + 0.877473 w_k1_MAT_s106
                                 + 1.13014 w_k1_MSFT_s106
                                 + 0.857722 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.835667 w_k1_ADM_s106
                                 + 0.59944 w_k1_AXP_s106
                                 + 0.634755 w_k1_DIS_s106
                                 + 0.717826 w_k1_FDX_s106
                                 + 0.823505 w_k1_IBM_s106
                                 + 0.703367 w_k1_INTC_s106
                                 + 0.892752 w_k1_KO_s106
                                 + 0.813046 w_k1_MAT_s106
                                 + 0.93004 w_k1_MSFT_s106
                                 + 0.801025 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.27205 w_k1_ADM_s106
                                 + 1.23396 w_k1_AXP_s106
                                 + 1.42967 w_k1_DIS_s106
                                 + 1.13501 w_k1_FDX_s106
                                 + 1.56765 w_k1_IBM_s106
                                 + 1.10011 w_k1_INTC_s106
                                 + 1.45601 w_k1_KO_s106 + 2.04107 w_k1_MAT_s106
                                 + 1.13214 w_k1_MSFT_s106
                                 + 1.12643 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.939919 w_k1_ADM_s106
                                 + 0.641173 w_k1_AXP_s106
                                 + 0.790217 w_k1_DIS_s106
                                 + 0.889283 w_k1_FDX_s106
                                 + 0.646023 w_k1_IBM_s106
                                 + 0.867814 w_k1_INTC_s106
                                 + 0.842535 w_k1_KO_s106
                                 + 0.869999 w_k1_MAT_s106
                                 + 1.09647 w_k1_MSFT_s106
                                 + 0.841245 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.978845 w_k1_ADM_s107
                                 + 0.602118 w_k1_AXP_s107
                                 + 0.77431 w_k1_DIS_s107
                                 + 0.821616 w_k1_FDX_s107
                                 + 0.656908 w_k1_IBM_s107
                                 + 0.762602 w_k1_INTC_s107
                                 + 0.878814 w_k1_KO_s107
                                 + 0.777177 w_k1_MAT_s107
                                 + 0.844467 w_k1_MSFT_s107
                                 + 0.900352 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.07856 w_k1_ADM_s107
                                 + 0.834343 w_k1_AXP_s107
                                 + 0.988058 w_k1_DIS_s107
                                 + 0.933329 w_k1_FDX_s107
                                 + 0.795762 w_k1_IBM_s107
                                 + 0.932566 w_k1_INTC_s107
                                 + 1.08461 w_k1_KO_s107
                                 + 0.786699 w_k1_MAT_s107
                                 + 0.933425 w_k1_MSFT_s107
                                 + 0.916997 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.28031 w_k1_ADM_s107
                                 + 0.797894 w_k1_AXP_s107
                                 + 0.777681 w_k1_DIS_s107
                                 + 1.00299 w_k1_FDX_s107
                                 + 0.578443 w_k1_IBM_s107
                                 + 0.986628 w_k1_INTC_s107
                                 + 0.76018 w_k1_KO_s107
                                 + 0.638398 w_k1_MAT_s107
                                 + 0.881669 w_k1_MSFT_s107
                                 + 0.996904 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.31859 w_k1_ADM_s107
                                 + 1.48666 w_k1_AXP_s107
                                 + 1.72531 w_k1_DIS_s107
                                 + 1.42984 w_k1_FDX_s107
                                 + 1.13582 w_k1_IBM_s107
                                 + 1.40518 w_k1_INTC_s107 + 1.5589 w_k1_KO_s107
                                 + 1.29782 w_k1_MAT_s107
                                 + 1.38389 w_k1_MSFT_s107
                                 + 1.36572 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.36706 w_k1_ADM_s107
                                 + 0.957223 w_k1_AXP_s107
                                 + 1.21866 w_k1_DIS_s107
                                 + 1.13214 w_k1_FDX_s107
                                 + 0.760854 w_k1_IBM_s107
                                 + 1.14275 w_k1_INTC_s107
                                 + 1.12177 w_k1_KO_s107
                                 + 0.687158 w_k1_MAT_s107
                                 + 1.01466 w_k1_MSFT_s107
                                 + 1.14067 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.866278 w_k1_ADM_s107
                                 + 1.01261 w_k1_AXP_s107 + 1.2936 w_k1_DIS_s107
                                 + 1.01752 w_k1_FDX_s107
                                 + 1.37539 w_k1_IBM_s107
                                 + 1.40915 w_k1_INTC_s107
                                 + 1.41304 w_k1_KO_s107 + 1.19354 w_k1_MAT_s107
                                 + 1.08362 w_k1_MSFT_s107 + 1.158 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.794887 w_k1_ADM_s107
                                 + 0.983899 w_k1_AXP_s107
                                 + 0.91186 w_k1_DIS_s107
                                 + 0.841938 w_k1_FDX_s107
                                 + 1.07915 w_k1_IBM_s107
                                 + 0.851162 w_k1_INTC_s107
                                 + 1.07393 w_k1_KO_s107 + 1.1497 w_k1_MAT_s107
                                 + 0.87865 w_k1_MSFT_s107
                                 + 0.99665 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.910503 w_k1_ADM_s107
                                 + 0.72156 w_k1_AXP_s107
                                 + 0.473775 w_k1_DIS_s107
                                 + 0.644634 w_k1_FDX_s107
                                 + 0.916978 w_k1_IBM_s107
                                 + 0.805356 w_k1_INTC_s107
                                 + 0.789263 w_k1_KO_s107
                                 + 0.806668 w_k1_MAT_s107
                                 + 0.722847 w_k1_MSFT_s107
                                 + 0.882466 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.04849 w_k1_ADM_s110
                                 + 1.27806 w_k1_AXP_s110
                                 + 1.19787 w_k1_DIS_s110
                                 + 1.20938 w_k1_FDX_s110
                                 + 0.879683 w_k1_IBM_s110
                                 + 1.39503 w_k1_INTC_s110
                                 + 1.00296 w_k1_KO_s110 + 1.24251 w_k1_MAT_s110
                                 + 1.1927 w_k1_MSFT_s110 + 1.14482 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.15598 w_k1_ADM_s110
                                 + 0.886562 w_k1_AXP_s110
                                 + 0.880344 w_k1_DIS_s110
                                 + 0.935452 w_k1_FDX_s110
                                 + 0.965929 w_k1_IBM_s110
                                 + 0.973114 w_k1_INTC_s110
                                 + 0.848548 w_k1_KO_s110
                                 + 0.686575 w_k1_MAT_s110
                                 + 0.959962 w_k1_MSFT_s110
                                 + 0.916306 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.954506 w_k1_ADM_s110
                                 + 0.864862 w_k1_AXP_s110
                                 + 0.962219 w_k1_DIS_s110
                                 + 0.865518 w_k1_FDX_s110
                                 + 0.76654 w_k1_IBM_s110
                                 + 0.802236 w_k1_INTC_s110
                                 + 0.741445 w_k1_KO_s110
                                 + 0.747264 w_k1_MAT_s110
                                 + 0.996128 w_k1_MSFT_s110
                                 + 1.00806 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.12451 w_k1_ADM_s110
                                 + 0.8238 w_k1_AXP_s110 + 1.09998 w_k1_DIS_s110
                                 + 1.15802 w_k1_FDX_s110
                                 + 0.777467 w_k1_IBM_s110
                                 + 1.13552 w_k1_INTC_s110
                                 + 0.822799 w_k1_KO_s110
                                 + 0.712243 w_k1_MAT_s110
                                 + 1.45692 w_k1_MSFT_s110
                                 + 0.954456 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.14418 w_k1_ADM_s110
                                 + 1.03539 w_k1_AXP_s110
                                 + 1.23696 w_k1_DIS_s110 + 1.0354 w_k1_FDX_s110
                                 + 1.32046 w_k1_IBM_s110
                                 + 1.17581 w_k1_INTC_s110
                                 + 1.18778 w_k1_KO_s110
                                 + 0.952722 w_k1_MAT_s110
                                 + 1.18908 w_k1_MSFT_s110
                                 + 1.18787 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.63116 w_k1_ADM_s110
                                 + 1.6352 w_k1_AXP_s110 + 1.74674 w_k1_DIS_s110
                                 + 1.39375 w_k1_FDX_s110
                                 + 1.14233 w_k1_IBM_s110
                                 + 1.31165 w_k1_INTC_s110
                                 + 1.12425 w_k1_KO_s110 + 1.25554 w_k1_MAT_s110
                                 + 1.32244 w_k1_MSFT_s110
                                 + 1.08181 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.859415 w_k1_ADM_s110
                                 + 0.802798 w_k1_AXP_s110
                                 + 0.6726 w_k1_DIS_s110
                                 + 0.761969 w_k1_FDX_s110
                                 + 0.996709 w_k1_IBM_s110
                                 + 0.936835 w_k1_INTC_s110
                                 + 0.830962 w_k1_KO_s110
                                 + 1.12371 w_k1_MAT_s110
                                 + 0.971386 w_k1_MSFT_s110
                                 + 0.896744 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.44162 w_k1_ADM_s110
                                 + 1.62605 w_k1_AXP_s110
                                 + 1.54421 w_k1_DIS_s110
                                 + 1.23417 w_k1_FDX_s110
                                 + 1.53019 w_k1_IBM_s110
                                 + 1.14393 w_k1_INTC_s110
                                 + 1.29889 w_k1_KO_s110 + 2.11002 w_k1_MAT_s110
                                 + 1.25921 w_k1_MSFT_s110 + 1.1949 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04187 w_k1_ADM_s111
                                 + 1.03943 w_k1_AXP_s111
                                 + 0.996321 w_k1_DIS_s111
                                 + 1.06049 w_k1_FDX_s111
                                 + 1.05816 w_k1_IBM_s111
                                 + 1.11719 w_k1_INTC_s111 + 1.0165 w_k1_KO_s111
                                 + 0.760748 w_k1_MAT_s111
                                 + 1.15388 w_k1_MSFT_s111
                                 + 1.06847 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.17224 w_k1_ADM_s111
                                 + 1.02027 w_k1_AXP_s111
                                 + 0.953473 w_k1_DIS_s111
                                 + 1.1922 w_k1_FDX_s111 + 0.76651 w_k1_IBM_s111
                                 + 1.25685 w_k1_INTC_s111
                                 + 0.874306 w_k1_KO_s111
                                 + 0.653015 w_k1_MAT_s111
                                 + 1.01766 w_k1_MSFT_s111
                                 + 1.13554 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.745213 w_k1_ADM_s111
                                 + 0.844096 w_k1_AXP_s111
                                 + 0.81864 w_k1_DIS_s111
                                 + 0.829915 w_k1_FDX_s111
                                 + 1.36118 w_k1_IBM_s111
                                 + 0.774695 w_k1_INTC_s111
                                 + 0.935441 w_k1_KO_s111
                                 + 1.18211 w_k1_MAT_s111
                                 + 0.844706 w_k1_MSFT_s111
                                 + 1.00723 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.91585 w_k1_ADM_s111
                                 + 1.04081 w_k1_AXP_s111
                                 + 1.26448 w_k1_DIS_s111
                                 + 1.17349 w_k1_FDX_s111
                                 + 0.932501 w_k1_IBM_s111
                                 + 1.15865 w_k1_INTC_s111
                                 + 1.05653 w_k1_KO_s111
                                 + 0.813084 w_k1_MAT_s111
                                 + 1.27838 w_k1_MSFT_s111
                                 + 1.13728 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.638061 w_k1_ADM_s111
                                 + 0.732113 w_k1_AXP_s111
                                 + 0.871778 w_k1_DIS_s111
                                 + 0.82419 w_k1_FDX_s111
                                 + 0.831093 w_k1_IBM_s111
                                 + 0.818623 w_k1_INTC_s111
                                 + 0.865128 w_k1_KO_s111
                                 + 0.768078 w_k1_MAT_s111
                                 + 0.935827 w_k1_MSFT_s111
                                 + 0.915697 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.851086 w_k1_ADM_s111
                                 + 0.654506 w_k1_AXP_s111
                                 + 0.556897 w_k1_DIS_s111
                                 + 0.807626 w_k1_FDX_s111
                                 + 0.946598 w_k1_IBM_s111
                                 + 1.02269 w_k1_INTC_s111
                                 + 0.834698 w_k1_KO_s111
                                 + 0.561206 w_k1_MAT_s111
                                 + 0.709309 w_k1_MSFT_s111
                                 + 0.889547 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.24856 w_k1_ADM_s111
                                 + 1.545 w_k1_AXP_s111 + 1.53084 w_k1_DIS_s111
                                 + 1.24235 w_k1_FDX_s111
                                 + 1.31238 w_k1_IBM_s111
                                 + 1.30779 w_k1_INTC_s111 + 1.3403 w_k1_KO_s111
                                 + 1.16383 w_k1_MAT_s111
                                 + 1.22347 w_k1_MSFT_s111
                                 + 1.43529 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.78773 w_k1_ADM_s111
                                 + 0.81933 w_k1_AXP_s111
                                 + 0.848765 w_k1_DIS_s111
                                 + 0.943132 w_k1_FDX_s111
                                 + 0.653114 w_k1_IBM_s111
                                 + 1.09119 w_k1_INTC_s111
                                 + 0.78559 w_k1_KO_s111
                                 + 0.530382 w_k1_MAT_s111
                                 + 0.904354 w_k1_MSFT_s111
                                 + 0.875393 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.889756 w_k1_ADM_s112
                                 + 0.813466 w_k1_AXP_s112
                                 + 0.806806 w_k1_DIS_s112
                                 + 0.963308 w_k1_FDX_s112
                                 + 0.721694 w_k1_IBM_s112
                                 + 0.765135 w_k1_INTC_s112
                                 + 0.749079 w_k1_KO_s112
                                 + 0.809787 w_k1_MAT_s112
                                 + 0.904067 w_k1_MSFT_s112
                                 + 0.886001 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.36446 w_k1_ADM_s112
                                 + 1.55217 w_k1_AXP_s112 + 1.3914 w_k1_DIS_s112
                                 + 1.2601 w_k1_FDX_s112 + 1.48618 w_k1_IBM_s112
                                 + 1.07789 w_k1_INTC_s112
                                 + 1.36739 w_k1_KO_s112 + 1.90562 w_k1_MAT_s112
                                 + 1.31775 w_k1_MSFT_s112
                                 + 1.19315 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.813663 w_k1_ADM_s112
                                 + 0.649827 w_k1_AXP_s112
                                 + 0.62888 w_k1_DIS_s112
                                 + 0.771587 w_k1_FDX_s112
                                 + 0.66696 w_k1_IBM_s112
                                 + 0.802137 w_k1_INTC_s112
                                 + 0.755061 w_k1_KO_s112
                                 + 0.630137 w_k1_MAT_s112
                                 + 0.785013 w_k1_MSFT_s112
                                 + 0.776148 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.993965 w_k1_ADM_s112
                                 + 0.887259 w_k1_AXP_s112
                                 + 0.990404 w_k1_DIS_s112
                                 + 0.881241 w_k1_FDX_s112
                                 + 1.42114 w_k1_IBM_s112
                                 + 0.710761 w_k1_INTC_s112
                                 + 1.18229 w_k1_KO_s112 + 1.73161 w_k1_MAT_s112
                                 + 1.03708 w_k1_MSFT_s112
                                 + 1.00127 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.978943 w_k1_ADM_s112
                                 + 1.15863 w_k1_AXP_s112
                                 + 0.849532 w_k1_DIS_s112
                                 + 0.885039 w_k1_FDX_s112
                                 + 1.05146 w_k1_IBM_s112
                                 + 0.754827 w_k1_INTC_s112
                                 + 0.995607 w_k1_KO_s112
                                 + 1.77731 w_k1_MAT_s112
                                 + 0.896017 w_k1_MSFT_s112
                                 + 1.05215 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.016 w_k1_ADM_s112
                                 + 0.793908 w_k1_AXP_s112
                                 + 0.893203 w_k1_DIS_s112
                                 + 0.964572 w_k1_FDX_s112
                                 + 1.10593 w_k1_IBM_s112
                                 + 0.977016 w_k1_INTC_s112
                                 + 1.00243 w_k1_KO_s112 + 1.05955 w_k1_MAT_s112
                                 + 1.10109 w_k1_MSFT_s112
                                 + 0.943562 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.85079 w_k1_ADM_s112
                                 + 0.672495 w_k1_AXP_s112
                                 + 0.764551 w_k1_DIS_s112
                                 + 0.757279 w_k1_FDX_s112
                                 + 0.921926 w_k1_IBM_s112
                                 + 0.642047 w_k1_INTC_s112
                                 + 0.837034 w_k1_KO_s112
                                 + 1.03157 w_k1_MAT_s112
                                 + 0.865095 w_k1_MSFT_s112
                                 + 0.871325 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.0371 w_k1_ADM_s112
                                 + 1.03773 w_k1_AXP_s112
                                 + 1.30862 w_k1_DIS_s112
                                 + 1.10625 w_k1_FDX_s112
                                 + 0.998515 w_k1_IBM_s112
                                 + 0.946841 w_k1_INTC_s112
                                 + 1.06067 w_k1_KO_s112 + 1.05329 w_k1_MAT_s112
                                 + 1.15475 w_k1_MSFT_s112 + 1.0384 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.826336 w_k1_ADM_s113
                                 + 1.07629 w_k1_AXP_s113
                                 + 1.23335 w_k1_DIS_s113
                                 + 1.16024 w_k1_FDX_s113
                                 + 0.829223 w_k1_IBM_s113
                                 + 1.14483 w_k1_INTC_s113
                                 + 1.01195 w_k1_KO_s113
                                 + 0.822522 w_k1_MAT_s113
                                 + 1.12565 w_k1_MSFT_s113
                                 + 0.912247 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.820529 w_k1_ADM_s113
                                 + 0.833137 w_k1_AXP_s113
                                 + 0.954746 w_k1_DIS_s113
                                 + 0.908363 w_k1_FDX_s113
                                 + 1.08101 w_k1_IBM_s113
                                 + 1.03528 w_k1_INTC_s113
                                 + 0.96036 w_k1_KO_s113
                                 + 0.980042 w_k1_MAT_s113
                                 + 0.97181 w_k1_MSFT_s113
                                 + 0.952868 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.11716 w_k1_ADM_s113
                                 + 1.34944 w_k1_AXP_s113
                                 + 1.36365 w_k1_DIS_s113
                                 + 1.17815 w_k1_FDX_s113
                                 + 1.55759 w_k1_IBM_s113
                                 + 1.26507 w_k1_INTC_s113
                                 + 1.23862 w_k1_KO_s113
                                 + 0.990345 w_k1_MAT_s113
                                 + 1.12362 w_k1_MSFT_s113
                                 + 1.04072 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.07032 w_k1_ADM_s113
                                 + 1.0553 w_k1_AXP_s113
                                 + 0.832036 w_k1_DIS_s113
                                 + 1.05246 w_k1_FDX_s113
                                 + 0.644868 w_k1_IBM_s113
                                 + 0.854003 w_k1_INTC_s113
                                 + 0.784489 w_k1_KO_s113
                                 + 0.791403 w_k1_MAT_s113
                                 + 0.908874 w_k1_MSFT_s113
                                 + 1.00051 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.12014 w_k1_ADM_s113
                                 + 1.07379 w_k1_AXP_s113
                                 + 0.857852 w_k1_DIS_s113
                                 + 1.01515 w_k1_FDX_s113
                                 + 0.854789 w_k1_IBM_s113
                                 + 0.87438 w_k1_INTC_s113
                                 + 0.815587 w_k1_KO_s113
                                 + 0.661179 w_k1_MAT_s113
                                 + 0.779992 w_k1_MSFT_s113
                                 + 0.92786 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.683895 w_k1_ADM_s113
                                 + 0.715748 w_k1_AXP_s113
                                 + 0.589856 w_k1_DIS_s113
                                 + 0.717628 w_k1_FDX_s113
                                 + 0.858946 w_k1_IBM_s113
                                 + 0.692927 w_k1_INTC_s113
                                 + 0.645397 w_k1_KO_s113
                                 + 0.745593 w_k1_MAT_s113
                                 + 0.671811 w_k1_MSFT_s113
                                 + 0.757453 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.32586 w_k1_ADM_s113
                                 + 1.15306 w_k1_AXP_s113
                                 + 1.10097 w_k1_DIS_s113
                                 + 1.14301 w_k1_FDX_s113
                                 + 1.22421 w_k1_IBM_s113
                                 + 1.02875 w_k1_INTC_s113
                                 + 1.26433 w_k1_KO_s113 + 1.15472 w_k1_MAT_s113
                                 + 0.993897 w_k1_MSFT_s113
                                 + 1.04722 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.641575 w_k1_ADM_s113
                                 + 0.700293 w_k1_AXP_s113
                                 + 0.705457 w_k1_DIS_s113
                                 + 0.894088 w_k1_FDX_s113
                                 + 0.535745 w_k1_IBM_s113
                                 + 0.912863 w_k1_INTC_s113
                                 + 0.576185 w_k1_KO_s113
                                 + 0.531308 w_k1_MAT_s113
                                 + 0.813839 w_k1_MSFT_s113
                                 + 0.797077 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.53778 w_k1_ADM_s114
                                 + 0.974892 w_k1_AXP_s114
                                 + 1.21817 w_k1_DIS_s114
                                 + 1.23462 w_k1_FDX_s114
                                 + 0.984061 w_k1_IBM_s114
                                 + 1.29998 w_k1_INTC_s114
                                 + 1.24019 w_k1_KO_s114
                                 + 0.677424 w_k1_MAT_s114
                                 + 1.32536 w_k1_MSFT_s114 + 1.2466 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.06553 w_k1_ADM_s114
                                 + 1.38612 w_k1_AXP_s114
                                 + 0.908764 w_k1_DIS_s114
                                 + 1.2271 w_k1_FDX_s114
                                 + 0.922585 w_k1_IBM_s114
                                 + 1.22425 w_k1_INTC_s114
                                 + 1.02186 w_k1_KO_s114
                                 + 0.978618 w_k1_MAT_s114
                                 + 1.09211 w_k1_MSFT_s114 + 1.0096 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.23966 w_k1_ADM_s114
                                 + 1.33626 w_k1_AXP_s114
                                 + 1.61948 w_k1_DIS_s114
                                 + 1.32047 w_k1_FDX_s114
                                 + 1.32837 w_k1_IBM_s114
                                 + 1.01637 w_k1_INTC_s114
                                 + 1.18609 w_k1_KO_s114 + 1.5916 w_k1_MAT_s114
                                 + 1.11308 w_k1_MSFT_s114
                                 + 1.08251 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.29387 w_k1_ADM_s114
                                 + 1.12496 w_k1_AXP_s114
                                 + 0.93959 w_k1_DIS_s114
                                 + 1.08894 w_k1_FDX_s114
                                 + 0.877517 w_k1_IBM_s114
                                 + 0.857558 w_k1_INTC_s114
                                 + 0.945487 w_k1_KO_s114
                                 + 0.873192 w_k1_MAT_s114
                                 + 0.907842 w_k1_MSFT_s114
                                 + 1.0197 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.755862 w_k1_ADM_s114
                                 + 0.820197 w_k1_AXP_s114
                                 + 0.683818 w_k1_DIS_s114
                                 + 0.854265 w_k1_FDX_s114
                                 + 0.72523 w_k1_IBM_s114
                                 + 0.995494 w_k1_INTC_s114
                                 + 0.754406 w_k1_KO_s114
                                 + 0.449172 w_k1_MAT_s114
                                 + 0.921405 w_k1_MSFT_s114
                                 + 0.802629 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.02488 w_k1_ADM_s114
                                 + 0.808267 w_k1_AXP_s114
                                 + 0.679026 w_k1_DIS_s114
                                 + 0.778853 w_k1_FDX_s114
                                 + 1.15931 w_k1_IBM_s114
                                 + 0.815981 w_k1_INTC_s114
                                 + 0.864241 w_k1_KO_s114
                                 + 1.00518 w_k1_MAT_s114
                                 + 0.837705 w_k1_MSFT_s114
                                 + 0.856439 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.13541 w_k1_ADM_s114
                                 + 0.885798 w_k1_AXP_s114
                                 + 0.947347 w_k1_DIS_s114
                                 + 0.902359 w_k1_FDX_s114
                                 + 1.23354 w_k1_IBM_s114
                                 + 0.958809 w_k1_INTC_s114
                                 + 1.11309 w_k1_KO_s114
                                 + 0.661255 w_k1_MAT_s114
                                 + 0.818216 w_k1_MSFT_s114
                                 + 0.966544 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.43022 w_k1_ADM_s114
                                 + 1.29888 w_k1_AXP_s114
                                 + 0.956032 w_k1_DIS_s114
                                 + 1.26895 w_k1_FDX_s114
                                 + 1.10578 w_k1_IBM_s114
                                 + 1.30191 w_k1_INTC_s114
                                 + 1.02571 w_k1_KO_s114
                                 + 0.975916 w_k1_MAT_s114
                                 + 0.998129 w_k1_MSFT_s114
                                 + 1.15947 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02077 w_k1_ADM_s115
                                 + 1.74001 w_k1_AXP_s115
                                 + 1.72438 w_k1_DIS_s115 + 1.5119 w_k1_FDX_s115
                                 + 1.13377 w_k1_IBM_s115
                                 + 1.63642 w_k1_INTC_s115
                                 + 1.20347 w_k1_KO_s115 + 1.09959 w_k1_MAT_s115
                                 + 1.37769 w_k1_MSFT_s115
                                 + 1.16995 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.998064 w_k1_ADM_s115
                                 + 1.19896 w_k1_AXP_s115
                                 + 0.927865 w_k1_DIS_s115
                                 + 1.15034 w_k1_FDX_s115
                                 + 0.970335 w_k1_IBM_s115
                                 + 1.23054 w_k1_INTC_s115
                                 + 1.01239 w_k1_KO_s115
                                 + 0.989912 w_k1_MAT_s115
                                 + 1.06368 w_k1_MSFT_s115
                                 + 1.00202 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.920336 w_k1_ADM_s115
                                 + 0.804282 w_k1_AXP_s115
                                 + 0.785797 w_k1_DIS_s115
                                 + 0.96645 w_k1_FDX_s115
                                 + 0.641076 w_k1_IBM_s115
                                 + 1.01517 w_k1_INTC_s115
                                 + 0.731678 w_k1_KO_s115
                                 + 0.706088 w_k1_MAT_s115
                                 + 0.936842 w_k1_MSFT_s115
                                 + 0.904578 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.03541 w_k1_ADM_s115
                                 + 0.829384 w_k1_AXP_s115
                                 + 1.16784 w_k1_DIS_s115
                                 + 0.949227 w_k1_FDX_s115
                                 + 1.04681 w_k1_IBM_s115
                                 + 1.07569 w_k1_INTC_s115
                                 + 1.14143 w_k1_KO_s115
                                 + 0.908631 w_k1_MAT_s115
                                 + 1.09257 w_k1_MSFT_s115
                                 + 1.04667 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.06874 w_k1_ADM_s115
                                 + 1.30866 w_k1_AXP_s115
                                 + 1.17558 w_k1_DIS_s115
                                 + 1.11233 w_k1_FDX_s115
                                 + 1.22331 w_k1_IBM_s115
                                 + 1.28532 w_k1_INTC_s115
                                 + 1.16442 w_k1_KO_s115 + 1.51198 w_k1_MAT_s115
                                 + 1.15351 w_k1_MSFT_s115
                                 + 1.13918 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.1711 w_k1_ADM_s115
                                 + 1.35593 w_k1_AXP_s115
                                 + 1.43922 w_k1_DIS_s115
                                 + 1.17969 w_k1_FDX_s115
                                 + 1.58446 w_k1_IBM_s115
                                 + 1.16925 w_k1_INTC_s115 + 1.5744 w_k1_KO_s115
                                 + 2.39215 w_k1_MAT_s115
                                 + 1.22625 w_k1_MSFT_s115
                                 + 1.27824 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.00468 w_k1_ADM_s115
                                 + 0.905512 w_k1_AXP_s115
                                 + 1.10225 w_k1_DIS_s115
                                 + 0.999096 w_k1_FDX_s115
                                 + 1.45829 w_k1_IBM_s115
                                 + 1.13457 w_k1_INTC_s115
                                 + 1.39194 w_k1_KO_s115 + 1.40577 w_k1_MAT_s115
                                 + 1.0777 w_k1_MSFT_s115 + 1.08117 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.873989 w_k1_ADM_s115
                                 + 0.884736 w_k1_AXP_s115
                                 + 0.761494 w_k1_DIS_s115
                                 + 0.79697 w_k1_FDX_s115 + 1.0312 w_k1_IBM_s115
                                 + 1.07147 w_k1_INTC_s115
                                 + 0.978851 w_k1_KO_s115
                                 + 0.936865 w_k1_MAT_s115
                                 + 0.985926 w_k1_MSFT_s115
                                 + 0.933439 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.836332 w_k1_ADM_s116
                                 + 1.00131 w_k1_AXP_s116
                                 + 0.855017 w_k1_DIS_s116
                                 + 0.796159 w_k1_FDX_s116
                                 + 1.5685 w_k1_IBM_s116
                                 + 0.982002 w_k1_INTC_s116
                                 + 1.30767 w_k1_KO_s116 + 1.53789 w_k1_MAT_s116
                                 + 0.868673 w_k1_MSFT_s116
                                 + 0.943941 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.20032 w_k1_ADM_s116
                                 + 1.03029 w_k1_AXP_s116
                                 + 1.09574 w_k1_DIS_s116
                                 + 1.06194 w_k1_FDX_s116
                                 + 1.30624 w_k1_IBM_s116
                                 + 1.15126 w_k1_INTC_s116
                                 + 1.37095 w_k1_KO_s116
                                 + 0.931058 w_k1_MAT_s116
                                 + 0.969936 w_k1_MSFT_s116
                                 + 1.14773 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.803212 w_k1_ADM_s116
                                 + 0.595111 w_k1_AXP_s116
                                 + 0.565051 w_k1_DIS_s116
                                 + 0.67867 w_k1_FDX_s116
                                 + 0.792191 w_k1_IBM_s116
                                 + 0.790393 w_k1_INTC_s116
                                 + 0.715183 w_k1_KO_s116
                                 + 0.496453 w_k1_MAT_s116
                                 + 0.715872 w_k1_MSFT_s116
                                 + 0.871387 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.04124 w_k1_ADM_s116
                                 + 1.12915 w_k1_AXP_s116
                                 + 0.984197 w_k1_DIS_s116
                                 + 1.10431 w_k1_FDX_s116
                                 + 0.901058 w_k1_IBM_s116
                                 + 1.15773 w_k1_INTC_s116
                                 + 1.04855 w_k1_KO_s116
                                 + 0.883516 w_k1_MAT_s116
                                 + 0.984754 w_k1_MSFT_s116
                                 + 0.999928 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.888841 w_k1_ADM_s116
                                 + 0.935544 w_k1_AXP_s116
                                 + 0.71021 w_k1_DIS_s116
                                 + 0.900306 w_k1_FDX_s116
                                 + 0.70324 w_k1_IBM_s116
                                 + 0.964841 w_k1_INTC_s116
                                 + 0.885846 w_k1_KO_s116
                                 + 0.745497 w_k1_MAT_s116
                                 + 0.934045 w_k1_MSFT_s116
                                 + 0.935641 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.1154 w_k1_ADM_s116
                                 + 1.52615 w_k1_AXP_s116
                                 + 1.19629 w_k1_DIS_s116
                                 + 1.23439 w_k1_FDX_s116
                                 + 1.23065 w_k1_IBM_s116
                                 + 1.40381 w_k1_INTC_s116
                                 + 1.43272 w_k1_KO_s116 + 1.28085 w_k1_MAT_s116
                                 + 1.18551 w_k1_MSFT_s116
                                 + 1.09904 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.1445 w_k1_ADM_s116
                                 + 1.82047 w_k1_AXP_s116
                                 + 1.46941 w_k1_DIS_s116 + 1.3341 w_k1_FDX_s116
                                 + 1.13597 w_k1_IBM_s116
                                 + 1.30463 w_k1_INTC_s116
                                 + 1.24519 w_k1_KO_s116 + 1.20453 w_k1_MAT_s116
                                 + 1.25733 w_k1_MSFT_s116 + 1.3517 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.764036 w_k1_ADM_s116
                                 + 0.854464 w_k1_AXP_s116
                                 + 0.810978 w_k1_DIS_s116
                                 + 0.814677 w_k1_FDX_s116
                                 + 1.11915 w_k1_IBM_s116
                                 + 0.856795 w_k1_INTC_s116
                                 + 1.07382 w_k1_KO_s116 + 1.09096 w_k1_MAT_s116
                                 + 0.811262 w_k1_MSFT_s116
                                 + 0.95426 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.828465 w_k1_ADM_s117
                                 + 1.00015 w_k1_AXP_s117
                                 + 0.760823 w_k1_DIS_s117
                                 + 0.957417 w_k1_FDX_s117
                                 + 0.629422 w_k1_IBM_s117
                                 + 0.799152 w_k1_INTC_s117
                                 + 0.666037 w_k1_KO_s117
                                 + 0.791006 w_k1_MAT_s117
                                 + 0.840576 w_k1_MSFT_s117
                                 + 0.805761 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.19195 w_k1_ADM_s117
                                 + 1.26032 w_k1_AXP_s117
                                 + 1.30623 w_k1_DIS_s117
                                 + 1.00517 w_k1_FDX_s117
                                 + 1.89555 w_k1_IBM_s117
                                 + 1.25741 w_k1_INTC_s117
                                 + 1.40664 w_k1_KO_s117 + 1.6485 w_k1_MAT_s117
                                 + 1.13897 w_k1_MSFT_s117
                                 + 1.17467 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.666637 w_k1_ADM_s117
                                 + 0.739634 w_k1_AXP_s117
                                 + 0.612525 w_k1_DIS_s117
                                 + 0.728942 w_k1_FDX_s117
                                 + 0.731738 w_k1_IBM_s117
                                 + 0.84054 w_k1_INTC_s117
                                 + 0.807649 w_k1_KO_s117
                                 + 0.728661 w_k1_MAT_s117
                                 + 0.682775 w_k1_MSFT_s117
                                 + 0.71462 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.903548 w_k1_ADM_s117
                                 + 1.57573 w_k1_AXP_s117
                                 + 1.24035 w_k1_DIS_s117
                                 + 1.21656 w_k1_FDX_s117
                                 + 0.931085 w_k1_IBM_s117
                                 + 1.28406 w_k1_INTC_s117
                                 + 1.07963 w_k1_KO_s117 + 1.31156 w_k1_MAT_s117
                                 + 1.27253 w_k1_MSFT_s117
                                 + 1.11073 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.684675 w_k1_ADM_s117
                                 + 1.12816 w_k1_AXP_s117
                                 + 1.11436 w_k1_DIS_s117
                                 + 0.954223 w_k1_FDX_s117
                                 + 1.20309 w_k1_IBM_s117
                                 + 1.03078 w_k1_INTC_s117
                                 + 1.04828 w_k1_KO_s117 + 1.52392 w_k1_MAT_s117
                                 + 1.12739 w_k1_MSFT_s117
                                 + 0.938119 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.866614 w_k1_ADM_s117
                                 + 1.29519 w_k1_AXP_s117
                                 + 1.56275 w_k1_DIS_s117
                                 + 1.15672 w_k1_FDX_s117
                                 + 1.25699 w_k1_IBM_s117
                                 + 1.25233 w_k1_INTC_s117 + 1.2183 w_k1_KO_s117
                                 + 1.32511 w_k1_MAT_s117
                                 + 1.27501 w_k1_MSFT_s117
                                 + 1.11552 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.777709 w_k1_ADM_s117
                                 + 0.967484 w_k1_AXP_s117
                                 + 0.788396 w_k1_DIS_s117
                                 + 0.842141 w_k1_FDX_s117
                                 + 0.96269 w_k1_IBM_s117
                                 + 0.835486 w_k1_INTC_s117
                                 + 0.924529 w_k1_KO_s117
                                 + 1.13708 w_k1_MAT_s117
                                 + 0.854067 w_k1_MSFT_s117
                                 + 0.838229 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.847161 w_k1_ADM_s117
                                 + 1.00115 w_k1_AXP_s117
                                 + 1.03175 w_k1_DIS_s117
                                 + 0.99331 w_k1_FDX_s117
                                 + 0.755145 w_k1_IBM_s117
                                 + 1.16101 w_k1_INTC_s117
                                 + 0.922195 w_k1_KO_s117
                                 + 0.784727 w_k1_MAT_s117
                                 + 0.875895 w_k1_MSFT_s117
                                 + 0.964164 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0231067491420892 w_k1_ADM_s000
                                 + 0.00979533473777189 w_k1_ADM_s001
                                 + 0.0113192497271752 w_k1_ADM_s002
                                 + 0.0104208855741862 w_k1_ADM_s003
                                 + 0.0466393752825098 w_k1_ADM_s004
                                 + 0.0305355793224064 w_k1_ADM_s005
                                 + 0.0225298142916674 w_k1_ADM_s006
                                 + 0.0110114703091609 w_k1_ADM_s007
                                 + 0.0444929330797467 w_k1_ADM_s010
                                 + 0.0650456772301535 w_k1_ADM_s011
                                 + 0.0347501116842364 w_k1_ADM_s012
                                 + 0.0655474264450025 w_k1_ADM_s013
                                 + 0.074937830315695 w_k1_ADM_s014
                                 + 0.100583450837985 w_k1_ADM_s015
                                 + 0.118495730349603 w_k1_ADM_s016
                                 + 0.0517730800971996 w_k1_ADM_s017
                                 + 0.0289177001333417 w_k1_ADM_s100
                                 + 0.0243112915358401 w_k1_ADM_s101
                                 + 0.013647627475366 w_k1_ADM_s102
                                 + 0.016364744218351 w_k1_ADM_s103
                                 + 0.02463182175367 w_k1_ADM_s104
                                 + 0.0280549476665812 w_k1_ADM_s105
                                 + 0.0187867536789641 w_k1_ADM_s106
                                 + 0.00877978202376306 w_k1_ADM_s107
                                 + 0.0129231819089264 w_k1_ADM_s110
                                 + 0.0189612962512043 w_k1_ADM_s111
                                 + 0.0180423068684715 w_k1_ADM_s112
                                 + 0.0118486149771037 w_k1_ADM_s113
                                 + 0.00602711288787407 w_k1_ADM_s114
                                 + 0.0181003503726383 w_k1_ADM_s115
                                 + 0.0251356388828751 w_k1_ADM_s116
                                 + 0.0110771070310331 w_k1_ADM_s117
                                 + 0.025253918598734 w_k1_AXP_s000
                                 + 0.0105261126985961 w_k1_AXP_s001
                                 + 0.0122336779361195 w_k1_AXP_s002
                                 + 0.00990038086918045 w_k1_AXP_s003
                                 + 0.0406073921938783 w_k1_AXP_s004
                                 + 0.0290942773087134 w_k1_AXP_s005
                                 + 0.0216017076270604 w_k1_AXP_s006
                                 + 0.0106978259895483 w_k1_AXP_s007
                                 + 0.0517828426039853 w_k1_AXP_s010
                                 + 0.0570733587345624 w_k1_AXP_s011
                                 + 0.0311766374284973 w_k1_AXP_s012
                                 + 0.0789049204738795 w_k1_AXP_s013
                                 + 0.0847642682228644 w_k1_AXP_s014
                                 + 0.103537389944589 w_k1_AXP_s015
                                 + 0.112009287930285 w_k1_AXP_s016
                                 + 0.0513813885241545 w_k1_AXP_s017
                                 + 0.0333799964594598 w_k1_AXP_s100
                                 + 0.0302932055936943 w_k1_AXP_s101
                                 + 0.0162371441517235 w_k1_AXP_s102
                                 + 0.0162198038065158 w_k1_AXP_s103
                                 + 0.025795958206315 w_k1_AXP_s104
                                 + 0.0289620606222748 w_k1_AXP_s105
                                 + 0.0173021935184471 w_k1_AXP_s106
                                 + 0.00843809432948051 w_k1_AXP_s107
                                 + 0.0119244170340456 w_k1_AXP_s110
                                 + 0.0209680380228729 w_k1_AXP_s111
                                 + 0.0171836196946855 w_k1_AXP_s112
                                 + 0.0126048706550433 w_k1_AXP_s113
                                 + 0.00527908237074209 w_k1_AXP_s114
                                 + 0.0193776558173459 w_k1_AXP_s115
                                 + 0.027791960405433 w_k1_AXP_s116
                                 + 0.0156522562140043 w_k1_AXP_s117
                                 + 0.0237124644564891 w_k1_DIS_s000
                                 + 0.00955499787647238 w_k1_DIS_s001
                                 + 0.0136607951300022 w_k1_DIS_s002
                                 + 0.0102411428087352 w_k1_DIS_s003
                                 + 0.044839613002307 w_k1_DIS_s004
                                 + 0.0330645155072876 w_k1_DIS_s005
                                 + 0.0231925407414182 w_k1_DIS_s006
                                 + 0.0110086544583113 w_k1_DIS_s007
                                 + 0.0478731241691043 w_k1_DIS_s010
                                 + 0.0613663465293745 w_k1_DIS_s011
                                 + 0.0319393317844386 w_k1_DIS_s012
                                 + 0.0735178875449261 w_k1_DIS_s013
                                 + 0.0838236843213217 w_k1_DIS_s014
                                 + 0.0993344371105741 w_k1_DIS_s015
                                 + 0.110924131894084 w_k1_DIS_s016
                                 + 0.0504327070018677 w_k1_DIS_s017
                                 + 0.0332402819338059 w_k1_DIS_s100
                                 + 0.0276054655611373 w_k1_DIS_s101
                                 + 0.0148930692988777 w_k1_DIS_s102
                                 + 0.0187069625157681 w_k1_DIS_s103
                                 + 0.0280919434726416 w_k1_DIS_s104
                                 + 0.0256128692166873 w_k1_DIS_s105
                                 + 0.0186333044567785 w_k1_DIS_s106
                                 + 0.00943108840503646 w_k1_DIS_s107
                                 + 0.0124766833490679 w_k1_DIS_s110
                                 + 0.0217257849149031 w_k1_DIS_s111
                                 + 0.018214392335649 w_k1_DIS_s112
                                 + 0.0124668316661221 w_k1_DIS_s113
                                 + 0.00512896359817602 w_k1_DIS_s114
                                 + 0.0187704850548474 w_k1_DIS_s115
                                 + 0.024056158809067 w_k1_DIS_s116
                                 + 0.0149020788129323 w_k1_DIS_s117
                                 + 0.0230553112651526 w_k1_FDX_s000
                                 + 0.00942994529205758 w_k1_FDX_s001
                                 + 0.0118927464252138 w_k1_FDX_s002
                                 + 0.010062817857919 w_k1_FDX_s003
                                 + 0.0434711794789105 w_k1_FDX_s004
                                 + 0.030887421507203 w_k1_FDX_s005
                                 + 0.0231020116077373 w_k1_FDX_s006
                                 + 0.0109049633394914 w_k1_FDX_s007
                                 + 0.0465741379282425 w_k1_FDX_s010
                                 + 0.0619289977335195 w_k1_FDX_s011
                                 + 0.0309963937910723 w_k1_FDX_s012
                                 + 0.07595824380889 w_k1_FDX_s013
                                 + 0.0787200648995367 w_k1_FDX_s014
                                 + 0.100642725187728 w_k1_FDX_s015
                                 + 0.107294409116345 w_k1_FDX_s016
                                 + 0.0503869963170133 w_k1_FDX_s017
                                 + 0.032133623439637 w_k1_FDX_s100
                                 + 0.0293965764383028 w_k1_FDX_s101
                                 + 0.0151408264025575 w_k1_FDX_s102
                                 + 0.0170803658798478 w_k1_FDX_s103
                                 + 0.0264751478867065 w_k1_FDX_s104
                                 + 0.0268433577234558 w_k1_FDX_s105
                                 + 0.0180408102195205 w_k1_FDX_s106
                                 + 0.00857290882398847 w_k1_FDX_s107
                                 + 0.0119677380567479 w_k1_FDX_s110
                                 + 0.020802876077691 w_k1_FDX_s111
                                 + 0.0174978638346346 w_k1_FDX_s112
                                 + 0.0126659497129962 w_k1_FDX_s113
                                 + 0.00542602348444207 w_k1_FDX_s114
                                 + 0.0189418334715448 w_k1_FDX_s115
                                 + 0.0255895207894362 w_k1_FDX_s116
                                 + 0.0135269122667421 w_k1_FDX_s117
                                 + 0.0235569797223238 w_k1_IBM_s000
                                 + 0.0096428627318192 w_k1_IBM_s001
                                 + 0.0126160389864436 w_k1_IBM_s002
                                 + 0.0101745078062232 w_k1_IBM_s003
                                 + 0.0410827127094279 w_k1_IBM_s004
                                 + 0.032300544801604 w_k1_IBM_s005
                                 + 0.0250695594184273 w_k1_IBM_s006
                                 + 0.0104891992456075 w_k1_IBM_s007
                                 + 0.0479540013586306 w_k1_IBM_s010
                                 + 0.0621639710565463 w_k1_IBM_s011
                                 + 0.033082168837194 w_k1_IBM_s012
                                 + 0.0762832269595456 w_k1_IBM_s013
                                 + 0.0776936804232705 w_k1_IBM_s014
                                 + 0.103975018954725 w_k1_IBM_s015
                                 + 0.110109966711117 w_k1_IBM_s016
                                 + 0.0489280559469014 w_k1_IBM_s017
                                 + 0.0329552599168502 w_k1_IBM_s100
                                 + 0.0283895975926229 w_k1_IBM_s101
                                 + 0.0160104139052154 w_k1_IBM_s102
                                 + 0.0170634718510849 w_k1_IBM_s103
                                 + 0.0254201418031993 w_k1_IBM_s104
                                 + 0.0275722106027297 w_k1_IBM_s105
                                 + 0.0183159769482261 w_k1_IBM_s106
                                 + 0.00903084261867337 w_k1_IBM_s107
                                 + 0.0116122847557455 w_k1_IBM_s110
                                 + 0.02080269536725 w_k1_IBM_s111
                                 + 0.0189393923313833 w_k1_IBM_s112
                                 + 0.0128785612675482 w_k1_IBM_s113
                                 + 0.00532555082902102 w_k1_IBM_s114
                                 + 0.0191456892445727 w_k1_IBM_s115
                                 + 0.0262097103768491 w_k1_IBM_s116
                                 + 0.0144332363388214 w_k1_IBM_s117
                                 + 0.0258792329111078 w_k1_INTC_s000
                                 + 0.00995213475505867 w_k1_INTC_s001
                                 + 0.0124174027755848 w_k1_INTC_s002
                                 + 0.0102183830544243 w_k1_INTC_s003
                                 + 0.0392542784054542 w_k1_INTC_s004
                                 + 0.0303074225269522 w_k1_INTC_s005
                                 + 0.0217018382082928 w_k1_INTC_s006
                                 + 0.0104309153632419 w_k1_INTC_s007
                                 + 0.0512764260749689 w_k1_INTC_s010
                                 + 0.0593559302662149 w_k1_INTC_s011
                                 + 0.0317103585224329 w_k1_INTC_s012
                                 + 0.081147887776306 w_k1_INTC_s013
                                 + 0.0768180705496512 w_k1_INTC_s014
                                 + 0.117496381699695 w_k1_INTC_s015
                                 + 0.114172534508877 w_k1_INTC_s016
                                 + 0.0404086105851231 w_k1_INTC_s017
                                 + 0.0338612126211563 w_k1_INTC_s100
                                 + 0.0296263524938463 w_k1_INTC_s101
                                 + 0.0152474380005607 w_k1_INTC_s102
                                 + 0.0174846014905558 w_k1_INTC_s103
                                 + 0.0265257288667169 w_k1_INTC_s104
                                 + 0.0290818872691481 w_k1_INTC_s105
                                 + 0.0173625437148837 w_k1_INTC_s106
                                 + 0.00953948687548581 w_k1_INTC_s107
                                 + 0.0126612871505532 w_k1_INTC_s110
                                 + 0.0216780063661684 w_k1_INTC_s111
                                 + 0.0152549578959098 w_k1_INTC_s112
                                 + 0.0126269044715828 w_k1_INTC_s113
                                 + 0.00532872040129793 w_k1_INTC_s114
                                 + 0.0212701586262392 w_k1_INTC_s115
                                 + 0.0275116877502408 w_k1_INTC_s116
                                 + 0.0146253389654296 w_k1_INTC_s117
                                 + 0.0254082854774204 w_k1_KO_s000
                                 + 0.01043087719482 w_k1_KO_s001
                                 + 0.012698622822691 w_k1_KO_s002
                                 + 0.00982160234838784 w_k1_KO_s003
                                 + 0.0414392978998434 w_k1_KO_s004
                                 + 0.0324424967909366 w_k1_KO_s005
                                 + 0.0221858351499241 w_k1_KO_s006
                                 + 0.0103292129362805 w_k1_KO_s007
                                 + 0.0448573931191566 w_k1_KO_s010
                                 + 0.0624354841451334 w_k1_KO_s011
                                 + 0.0317953945969179 w_k1_KO_s012
                                 + 0.0785308592911297 w_k1_KO_s013
                                 + 0.0833937759485626 w_k1_KO_s014
                                 + 0.098074776912099 w_k1_KO_s015
                                 + 0.112423418586857 w_k1_KO_s016
                                 + 0.04904868246739 w_k1_KO_s017
                                 + 0.0302157306648337 w_k1_KO_s100
                                 + 0.0271102605668673 w_k1_KO_s101
                                 + 0.0138110674662743 w_k1_KO_s102
                                 + 0.0156392590687223 w_k1_KO_s103
                                 + 0.0251464883614979 w_k1_KO_s104
                                 + 0.0275688623878953 w_k1_KO_s105
                                 + 0.0188347364282281 w_k1_KO_s106
                                 + 0.0101656061439291 w_k1_KO_s107
                                 + 0.0109385700071142 w_k1_KO_s110
                                 + 0.0206101571270029 w_k1_KO_s111
                                 + 0.0181132370838234 w_k1_KO_s112
                                 + 0.0119668086871244 w_k1_KO_s113
                                 + 0.00526222224328864 w_k1_KO_s114
                                 + 0.0195442204348725 w_k1_KO_s115
                                 + 0.0280323750442185 w_k1_KO_s116
                                 + 0.0140049240448638 w_k1_KO_s117
                                 + 0.027633872087274 w_k1_MAT_s000
                                 + 0.013625009572245 w_k1_MAT_s001
                                 + 0.0151297214487379 w_k1_MAT_s002
                                 + 0.0112789456843241 w_k1_MAT_s003
                                 + 0.0392891480219232 w_k1_MAT_s004
                                 + 0.0363026807201771 w_k1_MAT_s005
                                 + 0.0234008124313191 w_k1_MAT_s006
                                 + 0.010057414263006 w_k1_MAT_s007
                                 + 0.0462659477917314 w_k1_MAT_s010
                                 + 0.0592055279288342 w_k1_MAT_s011
                                 + 0.028099443316763 w_k1_MAT_s012
                                 + 0.0903285724711279 w_k1_MAT_s013
                                 + 0.0834617244021646 w_k1_MAT_s014
                                 + 0.0978895286192529 w_k1_MAT_s015
                                 + 0.0899705817449098 w_k1_MAT_s016
                                 + 0.0502417132331349 w_k1_MAT_s017
                                 + 0.0325440177792605 w_k1_MAT_s100
                                 + 0.0306763937317173 w_k1_MAT_s101
                                 + 0.0131547584452342 w_k1_MAT_s102
                                 + 0.0162099560922366 w_k1_MAT_s103
                                 + 0.029925412533171 w_k1_MAT_s104
                                 + 0.0275879409165291 w_k1_MAT_s105
                                 + 0.0225596351335705 w_k1_MAT_s106
                                 + 0.00888544032491892 w_k1_MAT_s107
                                 + 0.0114470972952887 w_k1_MAT_s110
                                 + 0.0176714651082332 w_k1_MAT_s111
                                 + 0.0220020831910873 w_k1_MAT_s112
                                 + 0.0109992926757822 w_k1_MAT_s113
                                 + 0.00428651839343739 w_k1_MAT_s114
                                 + 0.0204376310611672 w_k1_MAT_s115
                                 + 0.0245200832430141 w_k1_MAT_s116
                                 + 0.0166356160138979 w_k1_MAT_s117
                                 + 0.0261464715899581 w_k1_MSFT_s000
                                 + 0.0101919047601355 w_k1_MSFT_s001
                                 + 0.0134314345847364 w_k1_MSFT_s002
                                 + 0.00994174190841921 w_k1_MSFT_s003
                                 + 0.0409830183730261 w_k1_MSFT_s004
                                 + 0.0318229834734173 w_k1_MSFT_s005
                                 + 0.0227312666948962 w_k1_MSFT_s006
                                 + 0.010313577071647 w_k1_MSFT_s007
                                 + 0.0454845325989737 w_k1_MSFT_s010
                                 + 0.0616824470836828 w_k1_MSFT_s011
                                 + 0.0308140334497222 w_k1_MSFT_s012
                                 + 0.0747816496616075 w_k1_MSFT_s013
                                 + 0.0760133744555161 w_k1_MSFT_s014
                                 + 0.112071351761283 w_k1_MSFT_s015
                                 + 0.107852973637953 w_k1_MSFT_s016
                                 + 0.049116273649161 w_k1_MSFT_s017
                                 + 0.0303092066953734 w_k1_MSFT_s100
                                 + 0.0292654567399804 w_k1_MSFT_s101
                                 + 0.0144810012639257 w_k1_MSFT_s102
                                 + 0.0185536628568121 w_k1_MSFT_s103
                                 + 0.0284826945870535 w_k1_MSFT_s104
                                 + 0.0295574055829725 w_k1_MSFT_s105
                                 + 0.0195254022861826 w_k1_MSFT_s106
                                 + 0.0086589047969318 w_k1_MSFT_s107
                                 + 0.0132407984893227 w_k1_MSFT_s110
                                 + 0.021177310767317 w_k1_MSFT_s111
                                 + 0.0186410687681519 w_k1_MSFT_s112
                                 + 0.0118981706984486 w_k1_MSFT_s113
                                 + 0.00512089250118866 w_k1_MSFT_s114
                                 + 0.0196799059698157 w_k1_MSFT_s115
                                 + 0.0246744620566885 w_k1_MSFT_s116
                                 + 0.0142892440279874 w_k1_MSFT_s117
                                 + 0.0240146835230937 w_k1_PG_s000
                                 + 0.00995120670754578 w_k1_PG_s001
                                 + 0.0123349802941088 w_k1_PG_s002
                                 + 0.0099569422203732 w_k1_PG_s003
                                 + 0.045490290483208 w_k1_PG_s004
                                 + 0.0305842085019357 w_k1_PG_s005
                                 + 0.0247346570816298 w_k1_PG_s006
                                 + 0.0108995787507163 w_k1_PG_s007
                                 + 0.0468553278204318 w_k1_PG_s010
                                 + 0.0623062520090422 w_k1_PG_s011
                                 + 0.031400088402429 w_k1_PG_s012
                                 + 0.0720412335622908 w_k1_PG_s013
                                 + 0.0806959268636074 w_k1_PG_s014
                                 + 0.10826633989442 w_k1_PG_s015
                                 + 0.104206635726842 w_k1_PG_s016
                                 + 0.0508649596552466 w_k1_PG_s017
                                 + 0.0335954318665035 w_k1_PG_s100
                                 + 0.0273108613923252 w_k1_PG_s101
                                 + 0.0150809937930004 w_k1_PG_s102
                                 + 0.0170115770034891 w_k1_PG_s103
                                 + 0.024645650523925 w_k1_PG_s104
                                 + 0.0267819425021707 w_k1_PG_s105
                                 + 0.0168096664425791 w_k1_PG_s106
                                 + 0.00936325657727799 w_k1_PG_s107
                                 + 0.0120370229975801 w_k1_PG_s110
                                 + 0.02235985922434 w_k1_PG_s111
                                 + 0.017618280045154 w_k1_PG_s112
                                 + 0.0117844236091204 w_k1_PG_s113
                                 + 0.00519943615526935 w_k1_PG_s114
                                 + 0.0188560716231009 w_k1_PG_s115
                                 + 0.0263372997145044 w_k1_PG_s116
                                 + 0.0131586016876955 w_k1_PG_s117 >= 1.16
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
