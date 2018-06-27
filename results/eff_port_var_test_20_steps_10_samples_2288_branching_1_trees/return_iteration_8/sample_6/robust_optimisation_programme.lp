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
 w_k1_ADM_s0_return_constraint:  1.20572 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.964281 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.07107 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.948862 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.889953 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03193 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.695368 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.712037 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.08785 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.705045 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.992305 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.33975 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.747232 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.802595 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.754486 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.11292 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.991404 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.01047 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.848038 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.24431 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.05683 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.991222 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.846051 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.749306 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.15546 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.34144 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.782637 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.862163 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.43719 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.18454 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.28508 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.988425 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.421 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.921037 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.870767 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.775299 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.85675 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.10992 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20498 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.864709 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.11574 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.898745 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.875855 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.1915 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.558938 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.819131 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.40065 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.844107 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.934665 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.65651 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.847157 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.783526 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.689976 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.23801 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.898074 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.1935 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.770737 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.74448 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.16999 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.04216 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.933024 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.839299 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.16893 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.1215 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.691044 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.05256 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.44984 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.965536 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.45685 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.882079 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.49393 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.825653 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.11536 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.859756 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.852127 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.29644 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25566 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.811302 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.16898 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.957427 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.912139 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.29081 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.46459 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.708838 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.30892 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.823472 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.07353 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.87505 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.746461 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.670386 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.612629 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.06431 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.816989 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.28825 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.769128 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.6944 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.794068 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.920325 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.891834 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.835483 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.2774 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.32829 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.62267 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.871482 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.47721 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.946513 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.17932 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.759735 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.3961 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.812542 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.29644 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.808747 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.728844 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.58138 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.13868 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.898711 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.05326 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.989917 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.932657 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.1376 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.624944 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.728329 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.18308 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.969415 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.07918 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.65001 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.767741 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.791102 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.681181 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.08852 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.01994 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22231 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.840382 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.45758 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.00507 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.884519 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.850227 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.888263 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.03764 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.30904 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.747791 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.935329 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.37058 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.099 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.25752 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.949406 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.3824 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.896228 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.05095 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.780356 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.77286 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.23846 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17394 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.875807 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.29974 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.783412 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.800213 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.25994 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.73561 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.37309 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.35088 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.656591 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.844067 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.942813 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.04168 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.792192 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.958178 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.26711 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.6054 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.857379 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.805544 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.26295 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.87843 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.34935 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.26925 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.836609 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.73324 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.718501 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.888409 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.47649 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.19516 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.757367 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.2195 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.624721 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.797211 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.843416 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.82824 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.34294 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.0245 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.30792 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11134 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.897323 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.09937 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.966634 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.06017 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01023 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.599096 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.708795 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.18807 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.876982 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01704 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.38606 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.752898 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.775586 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.746128 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.981389 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.980986 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.3649 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.863174 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.30562 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.960315 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.87444 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.843833 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.935671 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.784351 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.07659 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.634214 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.846337 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.09769 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.930578 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.19261 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.14704 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.49492 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.994357 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.17188 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.01671 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.980635 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.44246 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21537 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.873126 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.17809 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.947429 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.929014 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.2434 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.618487 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.916566 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.14349 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.798487 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.956193 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.34664 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.815443 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.787131 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.7048 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.962219 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.662857 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.950247 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.726288 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.31659 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.763161 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.991769 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.00514 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.790811 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.33895 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.982771 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.740807 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00635 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.1575 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.864907 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.20423 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.723398 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.06493 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.87257 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.35208 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.913643 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.925296 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.42898 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28542 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.850201 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.26713 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.838449 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.693051 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.53435 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.566118 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.2797 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.62732 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.774403 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.726622 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.06326 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.971482 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.750483 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.694364 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.16137 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.80144 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.657636 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.650935 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.42534 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.889946 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.22762 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.36923 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.838718 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.11727 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.965981 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.717228 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.39824 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.53903 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.833984 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.36487 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.460287 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.718191 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.677683 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.52265 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.04523 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.922876 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.01803 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12492 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.895505 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.025 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.00059 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.949196 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.10599 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.625796 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.767992 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.14647 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.919332 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.09166 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.52914 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.806943 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.796464 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.672798 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.02631 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.962975 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.17481 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.874327 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.34489 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.863048 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.924005 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.952053 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.873795 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.05607 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.28263 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.761281 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.917809 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.18478 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.02292 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.21689 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.989506 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.29814 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.995139 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15236 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.794395 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.860913 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.3386 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11128 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.925106 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07996 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.981224 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.902551 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.13008 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.708991 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.968564 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.1151 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.917783 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06961 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26453 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.951132 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.925613 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.770357 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.14044 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.89278 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.0185 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.89601 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.15512 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.894036 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.944581 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.987811 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.950056 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.19381 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.03794 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.857256 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.10985 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.24972 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.02245 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.11964 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.806791 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.06985 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.916941 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.07699 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.00281 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.882742 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.17226 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.390999464450364 z_k1_s0000
                                 - 0.290000361208681 z_k1_s0001
                                 - 0.199000034194931 z_k1_s0002
                                 - 0.486999400878072 z_k1_s0003
                                 - 0.299000631873769 z_k1_s0004
                                 - 0.211000026248394 z_k1_s0005
                                 - 0.175999883450798 z_k1_s0006
                                 - 0.15999989404618 z_k1_s0007
                                 - 0.502000811930409 z_k1_s0010
                                 - 0.252001185790409 z_k1_s0011
                                 - 0.112000016350872 z_k1_s0012
                                 - 0.160000220329979 z_k1_s0013
                                 - 0.246001102842973 z_k1_s0014
                                 - 0.352001036246409 z_k1_s0015
                                 - 0.292000206772051 z_k1_s0016
                                 - 0.382001450983588 z_k1_s0017
                                 - 0.767000064151951 z_k1_s0020
                                 - 1.36499987006705 z_k1_s0021
                                 - 0.504000118201669 z_k1_s0022
                                 - 0.702000241791605 z_k1_s0023
                                 - 0.580999686197785 z_k1_s0024
                                 - 1.78800098512731 z_k1_s0025
                                 - 0.89900158674515 z_k1_s0026
                                 - 0.594999869506154 z_k1_s0027
                                 - 0.302999467383687 z_k1_s0030
                                 - 0.148000172412116 z_k1_s0031
                                 - 0.425001286859632 z_k1_s0032
                                 - 0.152000136017822 z_k1_s0033
                                 - 0.213000177755229 z_k1_s0034
                                 - 0.571999352618944 z_k1_s0035
                                 - 0.220999453966217 z_k1_s0036
                                 - 0.136000064594856 z_k1_s0037
                                 - 0.428002055353403 z_k1_s0040
                                 - 0.157000369745865 z_k1_s0041
                                 - 0.575001331094666 z_k1_s0042
                                 - 0.684002099954037 z_k1_s0043
                                 - 0.476000931837898 z_k1_s0044
                                 - 0.659002986952529 z_k1_s0045
                                 - 0.185000680231153 z_k1_s0046
                                 - 0.457999542151214 z_k1_s0047
                                 - 0.223000220452626 z_k1_s0050
                                 - 0.20100082113898 z_k1_s0051
                                 - 0.353000664124594 z_k1_s0052
                                 - 0.28700070918347 z_k1_s0053
                                 - 0.309000108497116 z_k1_s0054
                                 - 0.171000042983839 z_k1_s0055
                                 - 0.0730000193773116 z_k1_s0056
                                 - 0.140000058823613 z_k1_s0057
                                 - 0.281000788828471 z_k1_s0060
                                 - 0.21400052244245 z_k1_s0061
                                 - 0.283000889288027 z_k1_s0062
                                 - 0.639003276043464 z_k1_s0063
                                 - 0.221000564149985 z_k1_s0064
                                 - 0.246000580290794 z_k1_s0065
                                 - 0.484000138941552 z_k1_s0066
                                 - 0.731003558570294 z_k1_s0067
                                 - 0.393000464939748 z_k1_s0070
                                 - 0.177000179986572 z_k1_s0071
                                 - 0.160000248632023 z_k1_s0072
                                 - 0.506000748273732 z_k1_s0073
                                 - 0.257000608502916 z_k1_s0074
                                 - 0.20999962920377 z_k1_s0075
                                 - 0.1880002667926 z_k1_s0076
                                 - 0.137000067128511 z_k1_s0077
                                 - 0.660000379653834 z_k1_s0100
                                 - 0.57399891847481 z_k1_s0101
                                 - 1.0159997434187 z_k1_s0102
                                 - 0.359999846647575 z_k1_s0103
                                 - 0.292000062840796 z_k1_s0104
                                 - 0.432000113337066 z_k1_s0105
                                 - 0.767000411167413 z_k1_s0106
                                 - 0.854999140188029 z_k1_s0107
                                 - 0.946996955655338 z_k1_s0110
                                 - 0.552999084999879 z_k1_s0111
                                 - 0.189999498482994 z_k1_s0112
                                 - 0.442999695750843 z_k1_s0113
                                 - 0.301999138772114 z_k1_s0114
                                 - 0.263999174995713 z_k1_s0115
                                 - 0.277999009882223 z_k1_s0116
                                 - 0.226999176539847 z_k1_s0117
                                 - 0.204999899633103 z_k1_s0120
                                 - 0.081999959853241 z_k1_s0121
                                 - 0.359000149357482 z_k1_s0122
                                 - 0.266999899522138 z_k1_s0123
                                 - 0.104999944812077 z_k1_s0124
                                 - 0.205999649631313 z_k1_s0125
                                 - 0.236999649575831 z_k1_s0126
                                 - 0.0890000698407127 z_k1_s0127
                                 - 0.47699886293198 z_k1_s0130
                                 - 0.60300008830965 z_k1_s0131
                                 - 0.229999870245867 z_k1_s0132
                                 - 0.138999855661658 z_k1_s0133
                                 - 0.271000429424253 z_k1_s0134
                                 - 0.21399992322005 z_k1_s0135
                                 - 0.291999791320856 z_k1_s0136
                                 - 0.301000961847415 z_k1_s0137
                                 - 0.275999832340379 z_k1_s0140
                                 - 1.15999739418602 z_k1_s0141
                                 - 0.287999792459557 z_k1_s0142
                                 - 0.371999513293804 z_k1_s0143
                                 - 0.486999662068753 z_k1_s0144
                                 - 0.314999609027798 z_k1_s0145
                                 - 0.270999630324264 z_k1_s0146
                                 - 0.578997982050433 z_k1_s0147
                                 - 0.468999389730898 z_k1_s0150
                                 - 0.381999316124393 z_k1_s0151
                                 - 1.44599668305316 z_k1_s0152
                                 - 0.443999453924297 z_k1_s0153
                                 - 0.775000485000742 z_k1_s0154
                                 - 0.391999290447034 z_k1_s0155
                                 - 0.627000667025976 z_k1_s0156
                                 - 0.414999429388796 z_k1_s0157
                                 - 0.0999999734914006 z_k1_s0160
                                 - 0.149999960237101 z_k1_s0161
                                 - 0.179999952284521 z_k1_s0162
                                 - 0.145999961297445 z_k1_s0163
                                 - 0.239999936379362 z_k1_s0164
                                 - 0.0829999779978625 z_k1_s0165
                                 - 0.302999919678944 z_k1_s0166
                                 - 0.397999894495775 z_k1_s0167
                                 - 0.234999703890223 z_k1_s0170
                                 - 0.257999635256553 z_k1_s0171
                                 - 0.329999591811793 z_k1_s0172
                                 - 0.248999506287324 z_k1_s0173
                                 - 0.890999148771511 z_k1_s0174
                                 - 0.870000520374446 z_k1_s0175
                                 - 0.484000643775663 z_k1_s0176
                                 - 0.266999764225781 z_k1_s0177
                                 - 0.470998629279471 z_k1_s1000
                                 - 0.534996202001054 z_k1_s1001
                                 - 0.650998820657464 z_k1_s1002
                                 - 0.192999207070167 z_k1_s1003
                                 - 0.907995974747802 z_k1_s1004
                                 - 0.422998578245339 z_k1_s1005
                                 - 0.198999213449433 z_k1_s1006
                                 - 0.362998514452675 z_k1_s1007
                                 - 0.307000610402065 z_k1_s1010
                                 - 0.807002250786993 z_k1_s1011
                                 - 0.424000947570458 z_k1_s1012
                                 - 0.653000117524841 z_k1_s1013
                                 - 0.23900035521606 z_k1_s1014
                                 - 0.513002488293918 z_k1_s1015
                                 - 0.277000336922669 z_k1_s1016
                                 - 0.427002433720901 z_k1_s1017
                                 - 0.221000537921089 z_k1_s1020
                                 - 0.179000171370591 z_k1_s1021
                                 - 0.218999407800125 z_k1_s1022
                                 - 0.204999841949715 z_k1_s1023
                                 - 0.279999662004542 z_k1_s1024
                                 - 0.0849999629620644 z_k1_s1025
                                 - 0.215000485556732 z_k1_s1026
                                 - 0.265000365593283 z_k1_s1027
                                 - 0.468999698728777 z_k1_s1030
                                 - 0.275000196759505 z_k1_s1031
                                 - 0.201999768565605 z_k1_s1032
                                 - 0.226999863180016 z_k1_s1033
                                 - 0.344999364481143 z_k1_s1034
                                 - 0.260000224390759 z_k1_s1035
                                 - 0.190999718495275 z_k1_s1036
                                 - 0.140999740266206 z_k1_s1037
                                 - 0.825003090543766 z_k1_s1040
                                 - 0.264000172352947 z_k1_s1041
                                 - 1.0940039002796 z_k1_s1042
                                 - 0.498000603513874 z_k1_s1043
                                 - 0.600998101141048 z_k1_s1044
                                 - 0.844001196173135 z_k1_s1045
                                 - 0.746000836124309 z_k1_s1046
                                 - 0.96099966185133 z_k1_s1047
                                 - 0.228000727972937 z_k1_s1050
                                 - 0.122999850828057 z_k1_s1051
                                 - 0.188999717662678 z_k1_s1052
                                 - 0.211000465273413 z_k1_s1053
                                 - 0.341999111961421 z_k1_s1054
                                 - 0.218999657132961 z_k1_s1055
                                 - 0.272000243196421 z_k1_s1056
                                 - 0.395999597007325 z_k1_s1057
                                 - 0.371000093543953 z_k1_s1060
                                 - 0.571000042382698 z_k1_s1061
                                 - 0.473999925859845 z_k1_s1062
                                 - 0.610000550638477 z_k1_s1063
                                 - 1.28400278093808 z_k1_s1064
                                 - 1.04800100393957 z_k1_s1065
                                 - 0.597998763451378 z_k1_s1066
                                 - 0.93300056223709 z_k1_s1067
                                 - 0.400001115951469 z_k1_s1070
                                 - 0.31999881996146 z_k1_s1071
                                 - 0.418000790474337 z_k1_s1072
                                 - 0.182999972915371 z_k1_s1073
                                 - 0.190999943414408 z_k1_s1074
                                 - 0.251999848019543 z_k1_s1075
                                 - 0.20899987703724 z_k1_s1076
                                 - 0.61800005295025 z_k1_s1077
                                 - 0.301000362670007 z_k1_s1100
                                 - 0.293999943584477 z_k1_s1101
                                 - 0.334999875655755 z_k1_s1102
                                 - 0.367000109760294 z_k1_s1103
                                 - 0.109000135243293 z_k1_s1104
                                 - 0.108000103402516 z_k1_s1105
                                 - 0.19099999938585 z_k1_s1106
                                 - 0.257000531276483 z_k1_s1107
                                 - 0.250999496460607 z_k1_s1110
                                 - 0.203000154215547 z_k1_s1111
                                 - 0.237001216619372 z_k1_s1112
                                 - 0.435999441521436 z_k1_s1113
                                 - 0.350999466763757 z_k1_s1114
                                 - 0.34400032684314 z_k1_s1115
                                 - 0.148000170548814 z_k1_s1116
                                 - 0.175000162530665 z_k1_s1117
                                 - 0.359001997697649 z_k1_s1120
                                 - 0.488002268099662 z_k1_s1121
                                 - 0.538001031939776 z_k1_s1122
                                 - 0.352002294320552 z_k1_s1123
                                 - 0.534000760151011 z_k1_s1124
                                 - 0.228000973598661 z_k1_s1125
                                 - 0.41400187352696 z_k1_s1126
                                 - 0.17600083825898 z_k1_s1127
                                 - 0.782001454211508 z_k1_s1130
                                 - 0.499999975021635 z_k1_s1131
                                 - 1.07600177025016 z_k1_s1132
                                 - 0.396999894980197 z_k1_s1133
                                 - 0.380999775515531 z_k1_s1134
                                 - 0.773998888979728 z_k1_s1135
                                 - 0.739999261492175 z_k1_s1136
                                 - 0.360999876734644 z_k1_s1137
                                 - 0.240000464711363 z_k1_s1140
                                 - 0.0890000988262704 z_k1_s1141
                                 - 0.122000043986539 z_k1_s1142
                                 - 0.158999748108574 z_k1_s1143
                                 - 0.0960000501844958 z_k1_s1144
                                 - 0.156999529377567 z_k1_s1145
                                 - 0.320999824313657 z_k1_s1146
                                 - 0.172999922225124 z_k1_s1147
                                 - 0.513000577990429 z_k1_s1150
                                 - 0.159000465658799 z_k1_s1151
                                 - 0.510000406071799 z_k1_s1152
                                 - 0.317000874011389 z_k1_s1153
                                 - 0.212000620878399 z_k1_s1154
                                 - 0.214000735490819 z_k1_s1155
                                 - 0.525001265664948 z_k1_s1156
                                 - 0.432001700206478 z_k1_s1157
                                 - 0.391000876806515 z_k1_s1160
                                 - 0.369001317785309 z_k1_s1161
                                 - 0.352001086566978 z_k1_s1162
                                 - 0.821003024257601 z_k1_s1163
                                 - 0.562002462388389 z_k1_s1164
                                 - 0.187000692895351 z_k1_s1165
                                 - 0.399002466020176 z_k1_s1166
                                 - 0.620001030643378 z_k1_s1167
                                 - 0.250999968023511 z_k1_s1170
                                 - 0.342999565991822 z_k1_s1171
                                 - 0.0900001063782924 z_k1_s1172
                                 - 0.250999968023511 z_k1_s1173
                                 - 0.240999684070043 z_k1_s1174
                                 - 0.147000217711263 z_k1_s1175
                                 - 0.133000196976857 z_k1_s1176
                                 - 0.428000095594056 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.97245 w_k1_ADM_s000
                                 + 1.54109 w_k1_AXP_s000
                                 + 1.36015 w_k1_DIS_s000
                                 + 1.32996 w_k1_FDX_s000
                                 + 1.00076 w_k1_IBM_s000
                                 + 1.39537 w_k1_INTC_s000
                                 + 0.991549 w_k1_KO_s000
                                 + 1.23494 w_k1_MAT_s000
                                 + 1.33887 w_k1_MSFT_s000
                                 + 1.05569 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.664035 w_k1_ADM_s000
                                 + 0.89211 w_k1_AXP_s000
                                 + 0.742332 w_k1_DIS_s000
                                 + 0.784864 w_k1_FDX_s000
                                 + 1.1325 w_k1_IBM_s000
                                 + 1.08771 w_k1_INTC_s000
                                 + 0.833619 w_k1_KO_s000 + 1.1648 w_k1_MAT_s000
                                 + 0.897251 w_k1_MSFT_s000
                                 + 0.904369 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.914074 w_k1_ADM_s000
                                 + 0.914038 w_k1_AXP_s000
                                 + 1.22971 w_k1_DIS_s000
                                 + 1.13971 w_k1_FDX_s000
                                 + 0.807222 w_k1_IBM_s000
                                 + 1.28328 w_k1_INTC_s000
                                 + 1.11321 w_k1_KO_s000
                                 + 0.919509 w_k1_MAT_s000
                                 + 1.39688 w_k1_MSFT_s000
                                 + 1.11752 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.964695 w_k1_ADM_s000
                                 + 1.12459 w_k1_AXP_s000
                                 + 1.10162 w_k1_DIS_s000
                                 + 1.01425 w_k1_FDX_s000
                                 + 1.26866 w_k1_IBM_s000
                                 + 1.29595 w_k1_INTC_s000
                                 + 1.13065 w_k1_KO_s000 + 1.64424 w_k1_MAT_s000
                                 + 1.25296 w_k1_MSFT_s000
                                 + 1.05789 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.996101 w_k1_ADM_s000
                                 + 0.955114 w_k1_AXP_s000
                                 + 0.979578 w_k1_DIS_s000
                                 + 1.02795 w_k1_FDX_s000
                                 + 0.623212 w_k1_IBM_s000
                                 + 1.19747 w_k1_INTC_s000
                                 + 0.746555 w_k1_KO_s000 + 0.5474 w_k1_MAT_s000
                                 + 1.11407 w_k1_MSFT_s000
                                 + 0.949649 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.891511 w_k1_ADM_s000
                                 + 0.710033 w_k1_AXP_s000
                                 + 0.745702 w_k1_DIS_s000
                                 + 0.838138 w_k1_FDX_s000
                                 + 0.834942 w_k1_IBM_s000
                                 + 0.833006 w_k1_INTC_s000
                                 + 0.853773 w_k1_KO_s000
                                 + 0.782884 w_k1_MAT_s000
                                 + 0.973446 w_k1_MSFT_s000
                                 + 1.00162 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.662361 w_k1_ADM_s000
                                 + 0.672374 w_k1_AXP_s000
                                 + 0.562197 w_k1_DIS_s000
                                 + 0.697093 w_k1_FDX_s000
                                 + 0.655453 w_k1_IBM_s000
                                 + 0.813822 w_k1_INTC_s000
                                 + 0.642891 w_k1_KO_s000
                                 + 0.970654 w_k1_MAT_s000
                                 + 0.819015 w_k1_MSFT_s000
                                 + 0.778442 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.776355 w_k1_ADM_s000
                                 + 1.26268 w_k1_AXP_s000
                                 + 1.33387 w_k1_DIS_s000
                                 + 1.09322 w_k1_FDX_s000
                                 + 1.74958 w_k1_IBM_s000
                                 + 1.77732 w_k1_INTC_s000
                                 + 1.41123 w_k1_KO_s000 + 1.83651 w_k1_MAT_s000
                                 + 1.2868 w_k1_MSFT_s000 + 1.2576 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.938538 w_k1_ADM_s001
                                 + 0.768135 w_k1_AXP_s001
                                 + 0.931392 w_k1_DIS_s001
                                 + 0.817476 w_k1_FDX_s001
                                 + 1.04386 w_k1_IBM_s001
                                 + 0.918403 w_k1_INTC_s001
                                 + 1.00861 w_k1_KO_s001
                                 + 0.925134 w_k1_MAT_s001
                                 + 1.05648 w_k1_MSFT_s001
                                 + 0.820552 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.909728 w_k1_ADM_s001
                                 + 0.914069 w_k1_AXP_s001
                                 + 0.831357 w_k1_DIS_s001
                                 + 0.899707 w_k1_FDX_s001
                                 + 0.679786 w_k1_IBM_s001
                                 + 1.21399 w_k1_INTC_s001
                                 + 0.761031 w_k1_KO_s001
                                 + 0.54507 w_k1_MAT_s001
                                 + 1.0481 w_k1_MSFT_s001
                                 + 0.802403 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.30938 w_k1_ADM_s001
                                 + 1.66669 w_k1_AXP_s001
                                 + 1.97242 w_k1_DIS_s001 + 1.3836 w_k1_FDX_s001
                                 + 1.64835 w_k1_IBM_s001
                                 + 1.46363 w_k1_INTC_s001
                                 + 1.86352 w_k1_KO_s001 + 2.10567 w_k1_MAT_s001
                                 + 1.56194 w_k1_MSFT_s001
                                 + 1.25929 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.14841 w_k1_ADM_s001
                                 + 1.32935 w_k1_AXP_s001
                                 + 1.47466 w_k1_DIS_s001 + 1.2366 w_k1_FDX_s001
                                 + 0.947565 w_k1_IBM_s001
                                 + 1.51129 w_k1_INTC_s001
                                 + 1.21246 w_k1_KO_s001
                                 + 0.820838 w_k1_MAT_s001
                                 + 1.321 w_k1_MSFT_s001 + 1.11771 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.890974 w_k1_ADM_s001
                                 + 0.716715 w_k1_AXP_s001
                                 + 0.849252 w_k1_DIS_s001
                                 + 0.938137 w_k1_FDX_s001
                                 + 0.718065 w_k1_IBM_s001
                                 + 1.04771 w_k1_INTC_s001
                                 + 0.896658 w_k1_KO_s001
                                 + 0.733913 w_k1_MAT_s001
                                 + 1.0136 w_k1_MSFT_s001
                                 + 0.856796 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.18017 w_k1_ADM_s001
                                 + 1.2142 w_k1_AXP_s001 + 1.44281 w_k1_DIS_s001
                                 + 1.20946 w_k1_FDX_s001
                                 + 1.18998 w_k1_IBM_s001
                                 + 1.2323 w_k1_INTC_s001 + 1.36817 w_k1_KO_s001
                                 + 1.05818 w_k1_MAT_s001
                                 + 1.29168 w_k1_MSFT_s001
                                 + 1.03345 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05768 w_k1_ADM_s001
                                 + 1.15755 w_k1_AXP_s001
                                 + 1.30852 w_k1_DIS_s001
                                 + 1.01075 w_k1_FDX_s001
                                 + 1.55462 w_k1_IBM_s001
                                 + 1.04188 w_k1_INTC_s001
                                 + 1.31619 w_k1_KO_s001 + 1.33319 w_k1_MAT_s001
                                 + 1.19002 w_k1_MSFT_s001
                                 + 0.975589 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.946223 w_k1_ADM_s001
                                 + 1.05147 w_k1_AXP_s001
                                 + 0.922795 w_k1_DIS_s001
                                 + 0.976744 w_k1_FDX_s001
                                 + 0.865843 w_k1_IBM_s001
                                 + 1.0276 w_k1_INTC_s001 + 0.87059 w_k1_KO_s001
                                 + 1.1732 w_k1_MAT_s001
                                 + 0.930808 w_k1_MSFT_s001
                                 + 0.939232 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.822153 w_k1_ADM_s002
                                 + 0.747751 w_k1_AXP_s002
                                 + 0.830646 w_k1_DIS_s002
                                 + 0.838083 w_k1_FDX_s002
                                 + 0.726158 w_k1_IBM_s002
                                 + 0.7468 w_k1_INTC_s002
                                 + 0.869625 w_k1_KO_s002
                                 + 0.730757 w_k1_MAT_s002
                                 + 0.84652 w_k1_MSFT_s002
                                 + 0.888784 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.20383 w_k1_ADM_s002
                                 + 1.10427 w_k1_AXP_s002
                                 + 1.14506 w_k1_DIS_s002
                                 + 1.14676 w_k1_FDX_s002 + 1.0347 w_k1_IBM_s002
                                 + 1.00076 w_k1_INTC_s002
                                 + 1.21333 w_k1_KO_s002
                                 + 0.941141 w_k1_MAT_s002
                                 + 1.2027 w_k1_MSFT_s002 + 1.17267 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.646942 w_k1_ADM_s002
                                 + 0.56798 w_k1_AXP_s002
                                 + 0.617883 w_k1_DIS_s002
                                 + 0.683461 w_k1_FDX_s002
                                 + 0.839238 w_k1_IBM_s002
                                 + 0.716545 w_k1_INTC_s002
                                 + 0.776698 w_k1_KO_s002
                                 + 0.577532 w_k1_MAT_s002
                                 + 0.764176 w_k1_MSFT_s002
                                 + 0.855012 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.976855 w_k1_ADM_s002
                                 + 1.02529 w_k1_AXP_s002
                                 + 0.826759 w_k1_DIS_s002
                                 + 0.921045 w_k1_FDX_s002
                                 + 1.05824 w_k1_IBM_s002
                                 + 0.710895 w_k1_INTC_s002
                                 + 0.940383 w_k1_KO_s002
                                 + 1.21254 w_k1_MAT_s002
                                 + 0.81681 w_k1_MSFT_s002
                                 + 0.967815 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.829525 w_k1_ADM_s002
                                 + 0.847638 w_k1_AXP_s002
                                 + 0.568683 w_k1_DIS_s002
                                 + 0.772165 w_k1_FDX_s002
                                 + 0.936001 w_k1_IBM_s002
                                 + 0.787875 w_k1_INTC_s002
                                 + 0.761587 w_k1_KO_s002
                                 + 0.945659 w_k1_MAT_s002
                                 + 0.785737 w_k1_MSFT_s002
                                 + 0.819326 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.98717 w_k1_ADM_s002
                                 + 1.24296 w_k1_AXP_s002
                                 + 1.39271 w_k1_DIS_s002
                                 + 1.09173 w_k1_FDX_s002
                                 + 1.56061 w_k1_IBM_s002
                                 + 1.02544 w_k1_INTC_s002
                                 + 1.23223 w_k1_KO_s002 + 1.4893 w_k1_MAT_s002
                                 + 1.08107 w_k1_MSFT_s002
                                 + 1.12186 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.903003 w_k1_ADM_s002
                                 + 0.85777 w_k1_AXP_s002
                                 + 0.856345 w_k1_DIS_s002
                                 + 1.04471 w_k1_FDX_s002
                                 + 0.616001 w_k1_IBM_s002
                                 + 0.901138 w_k1_INTC_s002
                                 + 0.845914 w_k1_KO_s002
                                 + 0.659724 w_k1_MAT_s002
                                 + 0.947607 w_k1_MSFT_s002
                                 + 1.05365 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.26721 w_k1_ADM_s002
                                 + 1.23407 w_k1_AXP_s002
                                 + 1.69749 w_k1_DIS_s002
                                 + 1.43211 w_k1_FDX_s002
                                 + 0.825614 w_k1_IBM_s002
                                 + 0.994341 w_k1_INTC_s002
                                 + 1.23129 w_k1_KO_s002 + 0.9664 w_k1_MAT_s002
                                 + 1.38204 w_k1_MSFT_s002
                                 + 1.27745 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.867977 w_k1_ADM_s003
                                 + 0.925573 w_k1_AXP_s003
                                 + 1.15611 w_k1_DIS_s003
                                 + 0.925499 w_k1_FDX_s003
                                 + 1.15655 w_k1_IBM_s003
                                 + 1.03276 w_k1_INTC_s003
                                 + 1.02088 w_k1_KO_s003
                                 + 0.782047 w_k1_MAT_s003
                                 + 1.16595 w_k1_MSFT_s003
                                 + 0.981955 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.977862 w_k1_ADM_s003
                                 + 1.25037 w_k1_AXP_s003
                                 + 1.05888 w_k1_DIS_s003
                                 + 1.03973 w_k1_FDX_s003
                                 + 1.09974 w_k1_IBM_s003
                                 + 1.01141 w_k1_INTC_s003
                                 + 0.948939 w_k1_KO_s003
                                 + 1.02802 w_k1_MAT_s003
                                 + 1.05692 w_k1_MSFT_s003
                                 + 1.01707 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.07842 w_k1_ADM_s003
                                 + 1.03442 w_k1_AXP_s003
                                 + 0.941389 w_k1_DIS_s003
                                 + 1.02956 w_k1_FDX_s003
                                 + 0.811704 w_k1_IBM_s003
                                 + 0.976579 w_k1_INTC_s003
                                 + 0.870435 w_k1_KO_s003
                                 + 0.870602 w_k1_MAT_s003
                                 + 1.0865 w_k1_MSFT_s003
                                 + 0.927652 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.55235 w_k1_ADM_s003
                                 + 1.32502 w_k1_AXP_s003
                                 + 0.878929 w_k1_DIS_s003
                                 + 0.929977 w_k1_FDX_s003
                                 + 1.43712 w_k1_IBM_s003
                                 + 0.970873 w_k1_INTC_s003
                                 + 1.08342 w_k1_KO_s003 + 1.26433 w_k1_MAT_s003
                                 + 0.906112 w_k1_MSFT_s003
                                 + 1.1224 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.18422 w_k1_ADM_s003
                                 + 1.24839 w_k1_AXP_s003
                                 + 1.69121 w_k1_DIS_s003
                                 + 1.26041 w_k1_FDX_s003
                                 + 1.43264 w_k1_IBM_s003
                                 + 1.36718 w_k1_INTC_s003
                                 + 1.37604 w_k1_KO_s003 + 1.13869 w_k1_MAT_s003
                                 + 1.26749 w_k1_MSFT_s003
                                 + 1.26256 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.19187 w_k1_ADM_s003
                                 + 0.731769 w_k1_AXP_s003
                                 + 0.657191 w_k1_DIS_s003
                                 + 0.807012 w_k1_FDX_s003
                                 + 0.867645 w_k1_IBM_s003
                                 + 0.783716 w_k1_INTC_s003
                                 + 0.702058 w_k1_KO_s003
                                 + 0.57051 w_k1_MAT_s003
                                 + 0.814356 w_k1_MSFT_s003
                                 + 0.859506 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.84403 w_k1_ADM_s003
                                 + 1.66387 w_k1_AXP_s003
                                 + 1.41601 w_k1_DIS_s003
                                 + 1.44118 w_k1_FDX_s003
                                 + 1.10329 w_k1_IBM_s003
                                 + 1.34097 w_k1_INTC_s003
                                 + 1.14034 w_k1_KO_s003 + 1.1107 w_k1_MAT_s003
                                 + 1.36137 w_k1_MSFT_s003
                                 + 1.28015 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.29297 w_k1_ADM_s003
                                 + 1.29791 w_k1_AXP_s003
                                 + 0.881298 w_k1_DIS_s003
                                 + 0.976265 w_k1_FDX_s003
                                 + 1.09629 w_k1_IBM_s003
                                 + 0.89301 w_k1_INTC_s003
                                 + 0.89146 w_k1_KO_s003 + 1.17662 w_k1_MAT_s003
                                 + 1.01564 w_k1_MSFT_s003
                                 + 0.88666 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.882855 w_k1_ADM_s004
                                 + 1.31045 w_k1_AXP_s004
                                 + 1.19256 w_k1_DIS_s004
                                 + 1.00566 w_k1_FDX_s004
                                 + 1.54087 w_k1_IBM_s004
                                 + 0.999977 w_k1_INTC_s004
                                 + 1.15436 w_k1_KO_s004 + 2.43154 w_k1_MAT_s004
                                 + 1.00135 w_k1_MSFT_s004
                                 + 0.979435 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.992228 w_k1_ADM_s004
                                 + 0.948707 w_k1_AXP_s004
                                 + 1.60339 w_k1_DIS_s004
                                 + 1.11445 w_k1_FDX_s004
                                 + 0.869303 w_k1_IBM_s004
                                 + 0.893941 w_k1_INTC_s004
                                 + 1.12563 w_k1_KO_s004 + 1.12173 w_k1_MAT_s004
                                 + 1.18923 w_k1_MSFT_s004
                                 + 1.04752 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.632574 w_k1_ADM_s004
                                 + 0.98714 w_k1_AXP_s004
                                 + 0.681341 w_k1_DIS_s004
                                 + 0.78595 w_k1_FDX_s004
                                 + 0.872102 w_k1_IBM_s004
                                 + 0.892732 w_k1_INTC_s004
                                 + 0.866558 w_k1_KO_s004
                                 + 1.07792 w_k1_MAT_s004
                                 + 0.837506 w_k1_MSFT_s004
                                 + 0.888415 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.915798 w_k1_ADM_s004
                                 + 1.13048 w_k1_AXP_s004
                                 + 1.01397 w_k1_DIS_s004
                                 + 1.01049 w_k1_FDX_s004
                                 + 0.977138 w_k1_IBM_s004
                                 + 1.01605 w_k1_INTC_s004
                                 + 0.97214 w_k1_KO_s004 + 1.3343 w_k1_MAT_s004
                                 + 0.979286 w_k1_MSFT_s004
                                 + 0.957604 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.845412 w_k1_ADM_s004
                                 + 0.863288 w_k1_AXP_s004
                                 + 0.801987 w_k1_DIS_s004
                                 + 0.755336 w_k1_FDX_s004
                                 + 1.1703 w_k1_IBM_s004
                                 + 0.807987 w_k1_INTC_s004
                                 + 0.94173 w_k1_KO_s004 + 1.18871 w_k1_MAT_s004
                                 + 0.873543 w_k1_MSFT_s004
                                 + 0.824562 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.13545 w_k1_ADM_s004
                                 + 1.30464 w_k1_AXP_s004
                                 + 1.48652 w_k1_DIS_s004
                                 + 1.25404 w_k1_FDX_s004
                                 + 1.12815 w_k1_IBM_s004
                                 + 1.13497 w_k1_INTC_s004
                                 + 1.27489 w_k1_KO_s004 + 1.49013 w_k1_MAT_s004
                                 + 1.31077 w_k1_MSFT_s004
                                 + 0.988413 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.32849 w_k1_ADM_s004
                                 + 1.00097 w_k1_AXP_s004
                                 + 1.02146 w_k1_DIS_s004
                                 + 1.11407 w_k1_FDX_s004
                                 + 0.950746 w_k1_IBM_s004
                                 + 1.06463 w_k1_INTC_s004
                                 + 1.13337 w_k1_KO_s004 + 1.15665 w_k1_MAT_s004
                                 + 1.15242 w_k1_MSFT_s004
                                 + 1.10455 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.913871 w_k1_ADM_s004
                                 + 0.714937 w_k1_AXP_s004
                                 + 0.877609 w_k1_DIS_s004
                                 + 0.917156 w_k1_FDX_s004
                                 + 0.758232 w_k1_IBM_s004
                                 + 0.891822 w_k1_INTC_s004
                                 + 0.761155 w_k1_KO_s004
                                 + 0.531292 w_k1_MAT_s004
                                 + 0.94112 w_k1_MSFT_s004
                                 + 0.82353 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.696925 w_k1_ADM_s005
                                 + 0.59512 w_k1_AXP_s005
                                 + 0.488406 w_k1_DIS_s005
                                 + 0.686293 w_k1_FDX_s005
                                 + 0.646545 w_k1_IBM_s005
                                 + 0.683175 w_k1_INTC_s005
                                 + 0.535906 w_k1_KO_s005
                                 + 0.557679 w_k1_MAT_s005
                                 + 0.692926 w_k1_MSFT_s005
                                 + 0.686502 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.836373 w_k1_ADM_s005
                                 + 0.897644 w_k1_AXP_s005
                                 + 0.799001 w_k1_DIS_s005
                                 + 0.773369 w_k1_FDX_s005
                                 + 1.04673 w_k1_IBM_s005
                                 + 0.856542 w_k1_INTC_s005
                                 + 0.773508 w_k1_KO_s005
                                 + 1.01847 w_k1_MAT_s005
                                 + 0.901893 w_k1_MSFT_s005
                                 + 0.884423 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.39854 w_k1_ADM_s005
                                 + 1.32941 w_k1_AXP_s005
                                 + 1.44278 w_k1_DIS_s005
                                 + 1.19851 w_k1_FDX_s005
                                 + 1.63087 w_k1_IBM_s005
                                 + 1.05973 w_k1_INTC_s005
                                 + 1.30568 w_k1_KO_s005 + 1.91515 w_k1_MAT_s005
                                 + 1.07869 w_k1_MSFT_s005 + 1.1623 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.04709 w_k1_ADM_s005
                                 + 1.01735 w_k1_AXP_s005
                                 + 1.30828 w_k1_DIS_s005
                                 + 1.13446 w_k1_FDX_s005
                                 + 1.05809 w_k1_IBM_s005
                                 + 1.05593 w_k1_INTC_s005
                                 + 1.13135 w_k1_KO_s005 + 1.3331 w_k1_MAT_s005
                                 + 1.22227 w_k1_MSFT_s005
                                 + 1.19058 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.874327 w_k1_ADM_s005
                                 + 0.724202 w_k1_AXP_s005
                                 + 0.892864 w_k1_DIS_s005
                                 + 0.921871 w_k1_FDX_s005
                                 + 0.855085 w_k1_IBM_s005
                                 + 0.805432 w_k1_INTC_s005
                                 + 0.895243 w_k1_KO_s005
                                 + 0.697464 w_k1_MAT_s005
                                 + 0.891304 w_k1_MSFT_s005
                                 + 0.883074 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.08271 w_k1_ADM_s005
                                 + 1.15306 w_k1_AXP_s005
                                 + 0.847798 w_k1_DIS_s005
                                 + 0.925451 w_k1_FDX_s005
                                 + 1.20514 w_k1_IBM_s005
                                 + 1.03678 w_k1_INTC_s005
                                 + 0.995238 w_k1_KO_s005
                                 + 1.48539 w_k1_MAT_s005
                                 + 1.08336 w_k1_MSFT_s005
                                 + 1.03398 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.899607 w_k1_ADM_s005
                                 + 0.823898 w_k1_AXP_s005
                                 + 0.746515 w_k1_DIS_s005
                                 + 0.874587 w_k1_FDX_s005
                                 + 0.634758 w_k1_IBM_s005
                                 + 0.820101 w_k1_INTC_s005
                                 + 0.646542 w_k1_KO_s005
                                 + 0.933035 w_k1_MAT_s005
                                 + 0.856165 w_k1_MSFT_s005
                                 + 0.84808 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.05366 w_k1_ADM_s005
                                 + 1.19052 w_k1_AXP_s005
                                 + 0.89582 w_k1_DIS_s005
                                 + 1.05993 w_k1_FDX_s005
                                 + 0.759516 w_k1_IBM_s005
                                 + 1.29162 w_k1_INTC_s005
                                 + 0.809188 w_k1_KO_s005
                                 + 0.715363 w_k1_MAT_s005
                                 + 0.988419 w_k1_MSFT_s005
                                 + 0.886656 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.05877 w_k1_ADM_s006
                                 + 0.756133 w_k1_AXP_s006
                                 + 1.01764 w_k1_DIS_s006
                                 + 0.983954 w_k1_FDX_s006
                                 + 0.73853 w_k1_IBM_s006
                                 + 1.10915 w_k1_INTC_s006
                                 + 1.05247 w_k1_KO_s006
                                 + 0.696873 w_k1_MAT_s006
                                 + 1.01344 w_k1_MSFT_s006
                                 + 0.813447 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.32836 w_k1_ADM_s006
                                 + 1.11741 w_k1_AXP_s006 + 1.4759 w_k1_DIS_s006
                                 + 1.13677 w_k1_FDX_s006
                                 + 1.63813 w_k1_IBM_s006
                                 + 1.09368 w_k1_INTC_s006
                                 + 1.47401 w_k1_KO_s006 + 1.31467 w_k1_MAT_s006
                                 + 1.26671 w_k1_MSFT_s006
                                 + 1.14264 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.28719 w_k1_ADM_s006
                                 + 1.11559 w_k1_AXP_s006
                                 + 1.07448 w_k1_DIS_s006
                                 + 1.19127 w_k1_FDX_s006
                                 + 0.749617 w_k1_IBM_s006
                                 + 1.1359 w_k1_INTC_s006
                                 + 0.966922 w_k1_KO_s006
                                 + 0.727652 w_k1_MAT_s006
                                 + 1.27236 w_k1_MSFT_s006
                                 + 0.910339 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.967434 w_k1_ADM_s006
                                 + 0.931545 w_k1_AXP_s006
                                 + 1.06674 w_k1_DIS_s006
                                 + 1.02936 w_k1_FDX_s006
                                 + 1.13026 w_k1_IBM_s006
                                 + 1.12165 w_k1_INTC_s006
                                 + 1.27335 w_k1_KO_s006 + 1.06386 w_k1_MAT_s006
                                 + 1.15489 w_k1_MSFT_s006
                                 + 0.956655 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.894794 w_k1_ADM_s006
                                 + 0.794211 w_k1_AXP_s006
                                 + 0.729722 w_k1_DIS_s006
                                 + 0.827966 w_k1_FDX_s006
                                 + 1.11398 w_k1_IBM_s006
                                 + 1.20085 w_k1_INTC_s006
                                 + 1.01205 w_k1_KO_s006
                                 + 0.982054 w_k1_MAT_s006
                                 + 0.969748 w_k1_MSFT_s006
                                 + 0.913714 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.35615 w_k1_ADM_s006
                                 + 1.3079 w_k1_AXP_s006 + 1.77309 w_k1_DIS_s006
                                 + 1.48332 w_k1_FDX_s006 + 1.1768 w_k1_IBM_s006
                                 + 1.25033 w_k1_INTC_s006
                                 + 1.28437 w_k1_KO_s006 + 1.08675 w_k1_MAT_s006
                                 + 1.29851 w_k1_MSFT_s006
                                 + 1.12518 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.30162 w_k1_ADM_s006
                                 + 1.3508 w_k1_AXP_s006 + 1.07964 w_k1_DIS_s006
                                 + 1.19812 w_k1_FDX_s006 + 1.1459 w_k1_IBM_s006
                                 + 1.22125 w_k1_INTC_s006
                                 + 1.10845 w_k1_KO_s006 + 1.21662 w_k1_MAT_s006
                                 + 1.13272 w_k1_MSFT_s006
                                 + 0.98373 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.909744 w_k1_ADM_s006
                                 + 0.845056 w_k1_AXP_s006
                                 + 0.766491 w_k1_DIS_s006
                                 + 0.853414 w_k1_FDX_s006
                                 + 0.855056 w_k1_IBM_s006
                                 + 0.747321 w_k1_INTC_s006
                                 + 0.882049 w_k1_KO_s006
                                 + 0.929604 w_k1_MAT_s006
                                 + 0.861775 w_k1_MSFT_s006
                                 + 0.821529 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.930927 w_k1_ADM_s007
                                 + 0.987066 w_k1_AXP_s007
                                 + 0.816409 w_k1_DIS_s007
                                 + 0.864562 w_k1_FDX_s007
                                 + 1.13584 w_k1_IBM_s007
                                 + 1.18402 w_k1_INTC_s007
                                 + 1.06521 w_k1_KO_s007 + 1.10919 w_k1_MAT_s007
                                 + 1.02978 w_k1_MSFT_s007
                                 + 0.90681 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.979512 w_k1_ADM_s007
                                 + 1.29341 w_k1_AXP_s007
                                 + 1.38226 w_k1_DIS_s007
                                 + 1.33102 w_k1_FDX_s007
                                 + 1.59114 w_k1_IBM_s007
                                 + 1.13589 w_k1_INTC_s007 + 1.1776 w_k1_KO_s007
                                 + 1.73997 w_k1_MAT_s007
                                 + 1.25041 w_k1_MSFT_s007
                                 + 0.99402 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.863001 w_k1_ADM_s007
                                 + 0.559274 w_k1_AXP_s007
                                 + 0.637716 w_k1_DIS_s007
                                 + 0.747971 w_k1_FDX_s007
                                 + 0.689754 w_k1_IBM_s007
                                 + 0.941216 w_k1_INTC_s007
                                 + 0.783595 w_k1_KO_s007
                                 + 0.828534 w_k1_MAT_s007
                                 + 0.918368 w_k1_MSFT_s007
                                 + 0.811643 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.03421 w_k1_ADM_s007
                                 + 1.01461 w_k1_AXP_s007
                                 + 0.943504 w_k1_DIS_s007
                                 + 1.02769 w_k1_FDX_s007
                                 + 0.883518 w_k1_IBM_s007
                                 + 1.19913 w_k1_INTC_s007
                                 + 0.906107 w_k1_KO_s007
                                 + 0.830296 w_k1_MAT_s007
                                 + 1.17236 w_k1_MSFT_s007
                                 + 0.881156 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1383 w_k1_ADM_s007
                                 + 1.415 w_k1_AXP_s007 + 1.36626 w_k1_DIS_s007
                                 + 1.33774 w_k1_FDX_s007
                                 + 1.10286 w_k1_IBM_s007
                                 + 1.26084 w_k1_INTC_s007
                                 + 1.16178 w_k1_KO_s007 + 1.31182 w_k1_MAT_s007
                                 + 1.36697 w_k1_MSFT_s007
                                 + 1.17866 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.819117 w_k1_ADM_s007
                                 + 0.662704 w_k1_AXP_s007
                                 + 0.852432 w_k1_DIS_s007
                                 + 0.878606 w_k1_FDX_s007
                                 + 0.866369 w_k1_IBM_s007
                                 + 0.699382 w_k1_INTC_s007
                                 + 0.863017 w_k1_KO_s007
                                 + 0.962707 w_k1_MAT_s007
                                 + 0.944766 w_k1_MSFT_s007
                                 + 0.806397 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.809532 w_k1_ADM_s007
                                 + 1.17477 w_k1_AXP_s007
                                 + 1.07261 w_k1_DIS_s007
                                 + 1.04855 w_k1_FDX_s007
                                 + 1.45984 w_k1_IBM_s007
                                 + 1.05344 w_k1_INTC_s007
                                 + 1.25935 w_k1_KO_s007 + 1.6875 w_k1_MAT_s007
                                 + 1.10631 w_k1_MSFT_s007
                                 + 1.01293 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.08603 w_k1_ADM_s007
                                 + 1.23623 w_k1_AXP_s007
                                 + 1.15353 w_k1_DIS_s007 + 1.2848 w_k1_FDX_s007
                                 + 0.867076 w_k1_IBM_s007
                                 + 1.04284 w_k1_INTC_s007
                                 + 0.883667 w_k1_KO_s007
                                 + 0.805248 w_k1_MAT_s007
                                 + 1.26035 w_k1_MSFT_s007 + 1.0024 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.0046 w_k1_ADM_s010
                                 + 0.953123 w_k1_AXP_s010
                                 + 1.06792 w_k1_DIS_s010
                                 + 1.11885 w_k1_FDX_s010
                                 + 0.857319 w_k1_IBM_s010
                                 + 1.0337 w_k1_INTC_s010 + 1.08093 w_k1_KO_s010
                                 + 0.929235 w_k1_MAT_s010
                                 + 1.13428 w_k1_MSFT_s010
                                 + 1.06989 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.883986 w_k1_ADM_s010
                                 + 1.15353 w_k1_AXP_s010
                                 + 0.998144 w_k1_DIS_s010
                                 + 0.877671 w_k1_FDX_s010
                                 + 1.66244 w_k1_IBM_s010
                                 + 0.885898 w_k1_INTC_s010 + 1.149 w_k1_KO_s010
                                 + 1.54028 w_k1_MAT_s010
                                 + 0.92169 w_k1_MSFT_s010
                                 + 1.07045 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.69852 w_k1_ADM_s010
                                 + 0.706831 w_k1_AXP_s010
                                 + 0.684521 w_k1_DIS_s010
                                 + 0.752253 w_k1_FDX_s010
                                 + 0.822148 w_k1_IBM_s010
                                 + 0.984385 w_k1_INTC_s010
                                 + 0.763776 w_k1_KO_s010
                                 + 0.839065 w_k1_MAT_s010
                                 + 0.859045 w_k1_MSFT_s010
                                 + 0.852372 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.865638 w_k1_ADM_s010
                                 + 0.839737 w_k1_AXP_s010
                                 + 0.831098 w_k1_DIS_s010
                                 + 1.01185 w_k1_FDX_s010
                                 + 0.628396 w_k1_IBM_s010
                                 + 0.98465 w_k1_INTC_s010
                                 + 0.791851 w_k1_KO_s010
                                 + 0.701361 w_k1_MAT_s010
                                 + 0.937866 w_k1_MSFT_s010
                                 + 0.858283 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.22279 w_k1_ADM_s010
                                 + 1.65533 w_k1_AXP_s010
                                 + 1.22518 w_k1_DIS_s010
                                 + 1.30759 w_k1_FDX_s010
                                 + 1.11148 w_k1_IBM_s010
                                 + 1.11971 w_k1_INTC_s010
                                 + 1.26084 w_k1_KO_s010 + 1.86834 w_k1_MAT_s010
                                 + 1.30215 w_k1_MSFT_s010
                                 + 1.06625 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.12714 w_k1_ADM_s010
                                 + 1.05482 w_k1_AXP_s010
                                 + 0.811945 w_k1_DIS_s010
                                 + 0.97349 w_k1_FDX_s010
                                 + 0.949267 w_k1_IBM_s010
                                 + 0.733283 w_k1_INTC_s010
                                 + 0.846346 w_k1_KO_s010
                                 + 1.25164 w_k1_MAT_s010
                                 + 0.958291 w_k1_MSFT_s010
                                 + 1.04992 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.941433 w_k1_ADM_s010
                                 + 0.938122 w_k1_AXP_s010
                                 + 0.97561 w_k1_DIS_s010
                                 + 0.945347 w_k1_FDX_s010
                                 + 1.35277 w_k1_IBM_s010
                                 + 1.03471 w_k1_INTC_s010
                                 + 1.15984 w_k1_KO_s010
                                 + 0.951238 w_k1_MAT_s010
                                 + 1.13175 w_k1_MSFT_s010
                                 + 1.01559 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.16378 w_k1_ADM_s010
                                 + 1.43685 w_k1_AXP_s010
                                 + 1.50857 w_k1_DIS_s010
                                 + 1.25684 w_k1_FDX_s010 + 1.2378 w_k1_IBM_s010
                                 + 1.18659 w_k1_INTC_s010
                                 + 1.15448 w_k1_KO_s010 + 1.48531 w_k1_MAT_s010
                                 + 1.13064 w_k1_MSFT_s010
                                 + 1.13461 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.824312 w_k1_ADM_s011
                                 + 0.778099 w_k1_AXP_s011
                                 + 0.863219 w_k1_DIS_s011
                                 + 0.961896 w_k1_FDX_s011
                                 + 0.770361 w_k1_IBM_s011
                                 + 1.04831 w_k1_INTC_s011
                                 + 0.894704 w_k1_KO_s011
                                 + 0.754125 w_k1_MAT_s011
                                 + 1.06228 w_k1_MSFT_s011
                                 + 0.865936 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.85808 w_k1_ADM_s011
                                 + 0.817304 w_k1_AXP_s011
                                 + 0.952145 w_k1_DIS_s011
                                 + 0.862436 w_k1_FDX_s011
                                 + 1.08655 w_k1_IBM_s011
                                 + 0.928878 w_k1_INTC_s011
                                 + 1.06653 w_k1_KO_s011 + 1.19048 w_k1_MAT_s011
                                 + 0.862591 w_k1_MSFT_s011
                                 + 0.899166 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.988776 w_k1_ADM_s011
                                 + 1.18483 w_k1_AXP_s011
                                 + 1.23339 w_k1_DIS_s011
                                 + 1.20367 w_k1_FDX_s011
                                 + 0.757151 w_k1_IBM_s011
                                 + 1.39719 w_k1_INTC_s011
                                 + 1.12886 w_k1_KO_s011
                                 + 0.819413 w_k1_MAT_s011
                                 + 1.37414 w_k1_MSFT_s011
                                 + 1.07339 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.0191 w_k1_ADM_s011
                                 + 0.98585 w_k1_AXP_s011
                                 + 1.18997 w_k1_DIS_s011
                                 + 1.12941 w_k1_FDX_s011
                                 + 1.10647 w_k1_IBM_s011
                                 + 1.33677 w_k1_INTC_s011
                                 + 1.19582 w_k1_KO_s011 + 1.2789 w_k1_MAT_s011
                                 + 1.1498 w_k1_MSFT_s011
                                 + 0.954964 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.934538 w_k1_ADM_s011
                                 + 1.02084 w_k1_AXP_s011
                                 + 0.713209 w_k1_DIS_s011
                                 + 0.871803 w_k1_FDX_s011
                                 + 0.888112 w_k1_IBM_s011
                                 + 0.933136 w_k1_INTC_s011
                                 + 0.797557 w_k1_KO_s011
                                 + 1.34873 w_k1_MAT_s011
                                 + 0.845187 w_k1_MSFT_s011
                                 + 0.868064 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.750135 w_k1_ADM_s011
                                 + 0.929299 w_k1_AXP_s011
                                 + 0.882791 w_k1_DIS_s011
                                 + 0.881731 w_k1_FDX_s011
                                 + 1.5483 w_k1_IBM_s011 + 1.1573 w_k1_INTC_s011
                                 + 1.2262 w_k1_KO_s011 + 1.62268 w_k1_MAT_s011
                                 + 0.976361 w_k1_MSFT_s011
                                 + 1.02711 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.18511 w_k1_ADM_s011
                                 + 1.39651 w_k1_AXP_s011
                                 + 1.60498 w_k1_DIS_s011
                                 + 1.28153 w_k1_FDX_s011
                                 + 1.53193 w_k1_IBM_s011
                                 + 1.27558 w_k1_INTC_s011 + 1.5154 w_k1_KO_s011
                                 + 1.45047 w_k1_MAT_s011
                                 + 1.20047 w_k1_MSFT_s011
                                 + 1.10042 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.08823 w_k1_ADM_s011
                                 + 1.51949 w_k1_AXP_s011
                                 + 1.23715 w_k1_DIS_s011
                                 + 1.29667 w_k1_FDX_s011
                                 + 1.02304 w_k1_IBM_s011
                                 + 1.22072 w_k1_INTC_s011
                                 + 1.02411 w_k1_KO_s011 + 1.70935 w_k1_MAT_s011
                                 + 1.23084 w_k1_MSFT_s011
                                 + 1.01413 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.716786 w_k1_ADM_s012
                                 + 0.713385 w_k1_AXP_s012
                                 + 0.728439 w_k1_DIS_s012
                                 + 0.771104 w_k1_FDX_s012
                                 + 0.832119 w_k1_IBM_s012
                                 + 0.893681 w_k1_INTC_s012
                                 + 0.835914 w_k1_KO_s012
                                 + 0.469567 w_k1_MAT_s012
                                 + 0.832704 w_k1_MSFT_s012
                                 + 0.788706 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.668892 w_k1_ADM_s012
                                 + 0.964813 w_k1_AXP_s012
                                 + 1.03758 w_k1_DIS_s012 + 0.9622 w_k1_FDX_s012
                                 + 0.934107 w_k1_IBM_s012
                                 + 0.992348 w_k1_INTC_s012
                                 + 1.09168 w_k1_KO_s012
                                 + 0.662225 w_k1_MAT_s012
                                 + 1.06953 w_k1_MSFT_s012
                                 + 0.987116 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.13441 w_k1_ADM_s012
                                 + 1.1719 w_k1_AXP_s012 + 1.28918 w_k1_DIS_s012
                                 + 1.12906 w_k1_FDX_s012
                                 + 1.15872 w_k1_IBM_s012
                                 + 1.20444 w_k1_INTC_s012
                                 + 1.12997 w_k1_KO_s012
                                 + 0.546015 w_k1_MAT_s012
                                 + 1.17018 w_k1_MSFT_s012 + 1.1152 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.09236 w_k1_ADM_s012
                                 + 1.16804 w_k1_AXP_s012
                                 + 1.15364 w_k1_DIS_s012 + 1.1573 w_k1_FDX_s012
                                 + 0.76659 w_k1_IBM_s012
                                 + 0.879963 w_k1_INTC_s012
                                 + 1.00627 w_k1_KO_s012
                                 + 0.680287 w_k1_MAT_s012
                                 + 1.12997 w_k1_MSFT_s012
                                 + 1.00707 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.862213 w_k1_ADM_s012
                                 + 0.786386 w_k1_AXP_s012
                                 + 0.770156 w_k1_DIS_s012
                                 + 0.807013 w_k1_FDX_s012
                                 + 1.30019 w_k1_IBM_s012
                                 + 0.689796 w_k1_INTC_s012
                                 + 0.970238 w_k1_KO_s012
                                 + 0.77156 w_k1_MAT_s012
                                 + 0.703173 w_k1_MSFT_s012
                                 + 0.921764 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.979218 w_k1_ADM_s012
                                 + 1.21211 w_k1_AXP_s012
                                 + 0.981288 w_k1_DIS_s012
                                 + 0.952133 w_k1_FDX_s012
                                 + 1.13075 w_k1_IBM_s012
                                 + 0.977506 w_k1_INTC_s012
                                 + 1.18119 w_k1_KO_s012 + 1.04725 w_k1_MAT_s012
                                 + 1.00597 w_k1_MSFT_s012
                                 + 1.05998 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.855449 w_k1_ADM_s012
                                 + 0.890977 w_k1_AXP_s012
                                 + 0.618605 w_k1_DIS_s012
                                 + 0.846505 w_k1_FDX_s012
                                 + 0.781599 w_k1_IBM_s012
                                 + 0.930842 w_k1_INTC_s012
                                 + 0.889055 w_k1_KO_s012
                                 + 0.492912 w_k1_MAT_s012
                                 + 0.836263 w_k1_MSFT_s012
                                 + 0.828577 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.17019 w_k1_ADM_s012
                                 + 1.61622 w_k1_AXP_s012
                                 + 1.85721 w_k1_DIS_s012
                                 + 1.47936 w_k1_FDX_s012
                                 + 1.41005 w_k1_IBM_s012
                                 + 1.15122 w_k1_INTC_s012
                                 + 1.44902 w_k1_KO_s012 + 1.28439 w_k1_MAT_s012
                                 + 1.33142 w_k1_MSFT_s012
                                 + 1.32146 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.21388 w_k1_ADM_s013
                                 + 1.11845 w_k1_AXP_s013
                                 + 1.08861 w_k1_DIS_s013
                                 + 1.15163 w_k1_FDX_s013
                                 + 0.798714 w_k1_IBM_s013
                                 + 0.926311 w_k1_INTC_s013
                                 + 0.89259 w_k1_KO_s013 + 1.41053 w_k1_MAT_s013
                                 + 1.02779 w_k1_MSFT_s013
                                 + 0.976599 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.16302 w_k1_ADM_s013
                                 + 0.86981 w_k1_AXP_s013
                                 + 0.860112 w_k1_DIS_s013
                                 + 0.896827 w_k1_FDX_s013
                                 + 1.09523 w_k1_IBM_s013
                                 + 0.674653 w_k1_INTC_s013
                                 + 1.06487 w_k1_KO_s013 + 1.18859 w_k1_MAT_s013
                                 + 0.86764 w_k1_MSFT_s013
                                 + 0.995373 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.21762 w_k1_ADM_s013
                                 + 1.26178 w_k1_AXP_s013
                                 + 1.63134 w_k1_DIS_s013
                                 + 1.45698 w_k1_FDX_s013
                                 + 1.11498 w_k1_IBM_s013
                                 + 1.11204 w_k1_INTC_s013
                                 + 1.27278 w_k1_KO_s013 + 1.57839 w_k1_MAT_s013
                                 + 1.26523 w_k1_MSFT_s013
                                 + 1.04188 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.967608 w_k1_ADM_s013
                                 + 0.584757 w_k1_AXP_s013
                                 + 0.683553 w_k1_DIS_s013
                                 + 0.656554 w_k1_FDX_s013
                                 + 0.863171 w_k1_IBM_s013
                                 + 0.643625 w_k1_INTC_s013
                                 + 0.853292 w_k1_KO_s013
                                 + 0.917087 w_k1_MAT_s013
                                 + 0.771333 w_k1_MSFT_s013
                                 + 0.82252 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.03111 w_k1_ADM_s013
                                 + 0.97734 w_k1_AXP_s013
                                 + 0.83326 w_k1_DIS_s013
                                 + 0.978525 w_k1_FDX_s013
                                 + 0.72391 w_k1_IBM_s013
                                 + 0.706123 w_k1_INTC_s013
                                 + 0.746992 w_k1_KO_s013
                                 + 0.941738 w_k1_MAT_s013
                                 + 0.935623 w_k1_MSFT_s013
                                 + 0.920916 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.821932 w_k1_ADM_s013
                                 + 0.717263 w_k1_AXP_s013
                                 + 0.656274 w_k1_DIS_s013
                                 + 0.791572 w_k1_FDX_s013
                                 + 0.658601 w_k1_IBM_s013
                                 + 0.70474 w_k1_INTC_s013
                                 + 0.613467 w_k1_KO_s013
                                 + 0.620941 w_k1_MAT_s013
                                 + 0.723288 w_k1_MSFT_s013
                                 + 0.776547 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.09383 w_k1_ADM_s013
                                 + 1.2085 w_k1_AXP_s013 + 1.16138 w_k1_DIS_s013
                                 + 0.951534 w_k1_FDX_s013
                                 + 1.41354 w_k1_IBM_s013
                                 + 0.768308 w_k1_INTC_s013
                                 + 1.13259 w_k1_KO_s013 + 1.83326 w_k1_MAT_s013
                                 + 0.977902 w_k1_MSFT_s013
                                 + 1.08564 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.27943 w_k1_ADM_s013
                                 + 1.25392 w_k1_AXP_s013
                                 + 1.44603 w_k1_DIS_s013
                                 + 1.19983 w_k1_FDX_s013
                                 + 1.39293 w_k1_IBM_s013
                                 + 0.936734 w_k1_INTC_s013
                                 + 1.27042 w_k1_KO_s013 + 2.18668 w_k1_MAT_s013
                                 + 1.14214 w_k1_MSFT_s013
                                 + 1.21033 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.881617 w_k1_ADM_s014
                                 + 1.21233 w_k1_AXP_s014
                                 + 1.04957 w_k1_DIS_s014
                                 + 0.979233 w_k1_FDX_s014
                                 + 1.2725 w_k1_IBM_s014
                                 + 1.09286 w_k1_INTC_s014
                                 + 1.18757 w_k1_KO_s014 + 1.32539 w_k1_MAT_s014
                                 + 0.935059 w_k1_MSFT_s014
                                 + 0.865235 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.811155 w_k1_ADM_s014
                                 + 0.775226 w_k1_AXP_s014
                                 + 0.714295 w_k1_DIS_s014
                                 + 0.832646 w_k1_FDX_s014
                                 + 0.768505 w_k1_IBM_s014
                                 + 1.03998 w_k1_INTC_s014
                                 + 0.79755 w_k1_KO_s014
                                 + 0.667003 w_k1_MAT_s014
                                 + 0.817777 w_k1_MSFT_s014
                                 + 0.77851 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.01963 w_k1_ADM_s014
                                 + 1.08476 w_k1_AXP_s014
                                 + 1.24656 w_k1_DIS_s014 + 1.189 w_k1_FDX_s014
                                 + 0.945158 w_k1_IBM_s014
                                 + 1.22038 w_k1_INTC_s014
                                 + 1.17559 w_k1_KO_s014 + 1.06874 w_k1_MAT_s014
                                 + 1.18431 w_k1_MSFT_s014
                                 + 1.07773 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.18722 w_k1_ADM_s014
                                 + 1.02316 w_k1_AXP_s014
                                 + 0.939284 w_k1_DIS_s014
                                 + 1.09936 w_k1_FDX_s014
                                 + 0.779211 w_k1_IBM_s014
                                 + 0.974902 w_k1_INTC_s014
                                 + 0.902113 w_k1_KO_s014
                                 + 0.874386 w_k1_MAT_s014
                                 + 0.902661 w_k1_MSFT_s014
                                 + 0.947742 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.946821 w_k1_ADM_s014
                                 + 1.12987 w_k1_AXP_s014
                                 + 0.917455 w_k1_DIS_s014
                                 + 1.1332 w_k1_FDX_s014 + 1.05515 w_k1_IBM_s014
                                 + 1.16647 w_k1_INTC_s014
                                 + 1.00836 w_k1_KO_s014
                                 + 0.982365 w_k1_MAT_s014
                                 + 0.950219 w_k1_MSFT_s014
                                 + 0.968992 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.893566 w_k1_ADM_s014
                                 + 1.07028 w_k1_AXP_s014
                                 + 1.17048 w_k1_DIS_s014
                                 + 1.07233 w_k1_FDX_s014
                                 + 1.23629 w_k1_IBM_s014
                                 + 1.29804 w_k1_INTC_s014
                                 + 1.27622 w_k1_KO_s014 + 1.15771 w_k1_MAT_s014
                                 + 1.03808 w_k1_MSFT_s014
                                 + 1.02678 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.730794 w_k1_ADM_s014
                                 + 0.761509 w_k1_AXP_s014
                                 + 0.822745 w_k1_DIS_s014
                                 + 0.789913 w_k1_FDX_s014
                                 + 1.16397 w_k1_IBM_s014
                                 + 0.8617 w_k1_INTC_s014 + 1.091 w_k1_KO_s014
                                 + 1.21375 w_k1_MAT_s014
                                 + 0.862808 w_k1_MSFT_s014
                                 + 0.87127 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.17869 w_k1_ADM_s014
                                 + 1.40592 w_k1_AXP_s014
                                 + 1.60138 w_k1_DIS_s014 + 1.3789 w_k1_FDX_s014
                                 + 1.42888 w_k1_IBM_s014 + 1.323 w_k1_INTC_s014
                                 + 1.49291 w_k1_KO_s014 + 1.20588 w_k1_MAT_s014
                                 + 1.46974 w_k1_MSFT_s014
                                 + 1.26495 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.771436 w_k1_ADM_s015
                                 + 0.640333 w_k1_AXP_s015
                                 + 0.571437 w_k1_DIS_s015
                                 + 0.673993 w_k1_FDX_s015
                                 + 0.90383 w_k1_IBM_s015
                                 + 0.675176 w_k1_INTC_s015
                                 + 0.743745 w_k1_KO_s015
                                 + 1.06819 w_k1_MAT_s015
                                 + 0.746665 w_k1_MSFT_s015
                                 + 0.93314 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.988715 w_k1_ADM_s015
                                 + 1.0838 w_k1_AXP_s015 + 1.32284 w_k1_DIS_s015
                                 + 1.05835 w_k1_FDX_s015
                                 + 1.71193 w_k1_IBM_s015
                                 + 1.21216 w_k1_INTC_s015
                                 + 1.22025 w_k1_KO_s015 + 1.26757 w_k1_MAT_s015
                                 + 1.07874 w_k1_MSFT_s015
                                 + 1.09548 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.81017 w_k1_ADM_s015
                                 + 0.768913 w_k1_AXP_s015
                                 + 0.830416 w_k1_DIS_s015
                                 + 0.863539 w_k1_FDX_s015
                                 + 0.958798 w_k1_IBM_s015
                                 + 0.874185 w_k1_INTC_s015
                                 + 0.800324 w_k1_KO_s015
                                 + 0.82035 w_k1_MAT_s015
                                 + 0.844904 w_k1_MSFT_s015
                                 + 1.04734 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.11557 w_k1_ADM_s015
                                 + 0.801362 w_k1_AXP_s015
                                 + 0.66791 w_k1_DIS_s015
                                 + 0.905347 w_k1_FDX_s015
                                 + 0.623651 w_k1_IBM_s015
                                 + 0.902443 w_k1_INTC_s015
                                 + 0.579703 w_k1_KO_s015
                                 + 0.538664 w_k1_MAT_s015
                                 + 0.869628 w_k1_MSFT_s015
                                 + 0.925159 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.22865 w_k1_ADM_s015
                                 + 1.10607 w_k1_AXP_s015
                                 + 1.05266 w_k1_DIS_s015
                                 + 1.23281 w_k1_FDX_s015
                                 + 0.844308 w_k1_IBM_s015
                                 + 1.09016 w_k1_INTC_s015
                                 + 0.783803 w_k1_KO_s015
                                 + 0.90564 w_k1_MAT_s015
                                 + 1.06814 w_k1_MSFT_s015
                                 + 1.10514 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.30777 w_k1_ADM_s015
                                 + 1.16714 w_k1_AXP_s015
                                 + 1.68733 w_k1_DIS_s015
                                 + 1.41291 w_k1_FDX_s015
                                 + 1.07576 w_k1_IBM_s015
                                 + 1.31431 w_k1_INTC_s015
                                 + 1.20691 w_k1_KO_s015
                                 + 0.946912 w_k1_MAT_s015
                                 + 1.34258 w_k1_MSFT_s015
                                 + 1.25787 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.996102 w_k1_ADM_s015
                                 + 1.02641 w_k1_AXP_s015
                                 + 1.11092 w_k1_DIS_s015
                                 + 1.02185 w_k1_FDX_s015
                                 + 1.10862 w_k1_IBM_s015
                                 + 1.10324 w_k1_INTC_s015
                                 + 1.06746 w_k1_KO_s015
                                 + 0.988026 w_k1_MAT_s015
                                 + 1.05505 w_k1_MSFT_s015
                                 + 1.15967 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.64075 w_k1_ADM_s015
                                 + 1.67141 w_k1_AXP_s015
                                 + 1.35449 w_k1_DIS_s015
                                 + 1.33959 w_k1_FDX_s015
                                 + 1.33751 w_k1_IBM_s015
                                 + 1.45449 w_k1_INTC_s015
                                 + 1.21135 w_k1_KO_s015 + 1.5281 w_k1_MAT_s015
                                 + 1.37998 w_k1_MSFT_s015
                                 + 1.59989 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.687705 w_k1_ADM_s016
                                 + 0.857035 w_k1_AXP_s016
                                 + 0.955397 w_k1_DIS_s016
                                 + 0.960929 w_k1_FDX_s016
                                 + 0.879174 w_k1_IBM_s016
                                 + 1.11676 w_k1_INTC_s016
                                 + 0.93991 w_k1_KO_s016
                                 + 0.969149 w_k1_MAT_s016
                                 + 0.998558 w_k1_MSFT_s016
                                 + 0.963148 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.30437 w_k1_ADM_s016
                                 + 1.78595 w_k1_AXP_s016
                                 + 1.46673 w_k1_DIS_s016
                                 + 1.35778 w_k1_FDX_s016 + 1.3412 w_k1_IBM_s016
                                 + 1.41036 w_k1_INTC_s016
                                 + 1.47319 w_k1_KO_s016 + 1.86782 w_k1_MAT_s016
                                 + 1.35589 w_k1_MSFT_s016
                                 + 1.25766 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.833109 w_k1_ADM_s016
                                 + 0.953012 w_k1_AXP_s016
                                 + 1.01125 w_k1_DIS_s016
                                 + 0.939318 w_k1_FDX_s016
                                 + 1.2662 w_k1_IBM_s016 + 1.0076 w_k1_INTC_s016
                                 + 1.23624 w_k1_KO_s016 + 1.05492 w_k1_MAT_s016
                                 + 1.04283 w_k1_MSFT_s016
                                 + 0.981715 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.07582 w_k1_ADM_s016
                                 + 0.860175 w_k1_AXP_s016
                                 + 1.01671 w_k1_DIS_s016
                                 + 1.04637 w_k1_FDX_s016
                                 + 0.851708 w_k1_IBM_s016
                                 + 1.03883 w_k1_INTC_s016
                                 + 1.03766 w_k1_KO_s016
                                 + 0.576746 w_k1_MAT_s016
                                 + 1.01876 w_k1_MSFT_s016
                                 + 0.99455 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.698145 w_k1_ADM_s016
                                 + 0.77683 w_k1_AXP_s016
                                 + 0.738817 w_k1_DIS_s016
                                 + 0.821972 w_k1_FDX_s016
                                 + 0.867003 w_k1_IBM_s016
                                 + 0.97209 w_k1_INTC_s016
                                 + 0.827608 w_k1_KO_s016
                                 + 0.907351 w_k1_MAT_s016
                                 + 0.920981 w_k1_MSFT_s016
                                 + 0.853663 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.629847 w_k1_ADM_s016
                                 + 0.596749 w_k1_AXP_s016
                                 + 0.586552 w_k1_DIS_s016
                                 + 0.682736 w_k1_FDX_s016
                                 + 1.04959 w_k1_IBM_s016
                                 + 0.706459 w_k1_INTC_s016
                                 + 0.711376 w_k1_KO_s016
                                 + 0.727504 w_k1_MAT_s016
                                 + 0.627948 w_k1_MSFT_s016
                                 + 0.817263 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.23615 w_k1_ADM_s016
                                 + 1.13369 w_k1_AXP_s016
                                 + 1.44636 w_k1_DIS_s016
                                 + 1.25389 w_k1_FDX_s016
                                 + 1.27321 w_k1_IBM_s016 + 1.14 w_k1_INTC_s016
                                 + 1.36587 w_k1_KO_s016 + 1.05188 w_k1_MAT_s016
                                 + 1.32413 w_k1_MSFT_s016
                                 + 1.19514 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.899936 w_k1_ADM_s016
                                 + 1.09215 w_k1_AXP_s016
                                 + 1.05615 w_k1_DIS_s016
                                 + 0.95064 w_k1_FDX_s016
                                 + 1.02648 w_k1_IBM_s016
                                 + 1.12422 w_k1_INTC_s016
                                 + 0.952557 w_k1_KO_s016
                                 + 0.896489 w_k1_MAT_s016
                                 + 1.00315 w_k1_MSFT_s016
                                 + 1.01149 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.895568 w_k1_ADM_s017
                                 + 1.2639 w_k1_AXP_s017 + 1.02139 w_k1_DIS_s017
                                 + 1.12593 w_k1_FDX_s017 + 1.2812 w_k1_IBM_s017
                                 + 1.18649 w_k1_INTC_s017 + 1.1064 w_k1_KO_s017
                                 + 1.45975 w_k1_MAT_s017
                                 + 1.01109 w_k1_MSFT_s017
                                 + 1.06538 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.677703 w_k1_ADM_s017
                                 + 0.600093 w_k1_AXP_s017
                                 + 0.797912 w_k1_DIS_s017
                                 + 0.806158 w_k1_FDX_s017
                                 + 1.01884 w_k1_IBM_s017
                                 + 0.8976 w_k1_INTC_s017
                                 + 0.994114 w_k1_KO_s017
                                 + 0.872067 w_k1_MAT_s017
                                 + 0.899298 w_k1_MSFT_s017
                                 + 0.908329 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.13647 w_k1_ADM_s017
                                 + 1.20952 w_k1_AXP_s017
                                 + 1.11917 w_k1_DIS_s017
                                 + 1.06989 w_k1_FDX_s017
                                 + 1.72032 w_k1_IBM_s017
                                 + 1.19335 w_k1_INTC_s017 + 1.1624 w_k1_KO_s017
                                 + 1.35993 w_k1_MAT_s017
                                 + 1.00411 w_k1_MSFT_s017 + 1.0343 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.934999 w_k1_ADM_s017
                                 + 1.22043 w_k1_AXP_s017
                                 + 1.37003 w_k1_DIS_s017
                                 + 1.19922 w_k1_FDX_s017 + 1.4098 w_k1_IBM_s017
                                 + 1.18934 w_k1_INTC_s017 + 1.3588 w_k1_KO_s017
                                 + 1.20946 w_k1_MAT_s017
                                 + 1.18644 w_k1_MSFT_s017
                                 + 1.16806 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.999902 w_k1_ADM_s017
                                 + 1.13975 w_k1_AXP_s017
                                 + 1.20131 w_k1_DIS_s017
                                 + 1.15727 w_k1_FDX_s017
                                 + 0.882328 w_k1_IBM_s017
                                 + 1.05927 w_k1_INTC_s017
                                 + 1.02665 w_k1_KO_s017
                                 + 0.944414 w_k1_MAT_s017
                                 + 1.06989 w_k1_MSFT_s017
                                 + 1.10946 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.681265 w_k1_ADM_s017
                                 + 0.765772 w_k1_AXP_s017
                                 + 0.743296 w_k1_DIS_s017
                                 + 0.872849 w_k1_FDX_s017
                                 + 0.815309 w_k1_IBM_s017
                                 + 1.04552 w_k1_INTC_s017
                                 + 0.805141 w_k1_KO_s017
                                 + 0.766217 w_k1_MAT_s017
                                 + 0.852712 w_k1_MSFT_s017
                                 + 0.923527 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.986348 w_k1_ADM_s017
                                 + 0.913023 w_k1_AXP_s017
                                 + 0.929736 w_k1_DIS_s017
                                 + 0.899596 w_k1_FDX_s017
                                 + 1.14376 w_k1_IBM_s017
                                 + 1.19871 w_k1_INTC_s017
                                 + 1.00248 w_k1_KO_s017
                                 + 0.964634 w_k1_MAT_s017
                                 + 0.982023 w_k1_MSFT_s017
                                 + 0.97957 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.882829 w_k1_ADM_s017
                                 + 0.663037 w_k1_AXP_s017
                                 + 0.695662 w_k1_DIS_s017
                                 + 0.87196 w_k1_FDX_s017
                                 + 0.599637 w_k1_IBM_s017
                                 + 0.969916 w_k1_INTC_s017
                                 + 0.832714 w_k1_KO_s017
                                 + 0.592484 w_k1_MAT_s017
                                 + 0.825855 w_k1_MSFT_s017
                                 + 0.923791 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.44199 w_k1_ADM_s100
                                 + 1.29281 w_k1_AXP_s100 + 1.3682 w_k1_DIS_s100
                                 + 1.34386 w_k1_FDX_s100
                                 + 0.807943 w_k1_IBM_s100
                                 + 1.14508 w_k1_INTC_s100
                                 + 1.04063 w_k1_KO_s100
                                 + 0.805122 w_k1_MAT_s100
                                 + 1.111 w_k1_MSFT_s100 + 1.0866 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.940089 w_k1_ADM_s100
                                 + 0.803122 w_k1_AXP_s100
                                 + 0.781428 w_k1_DIS_s100
                                 + 0.804606 w_k1_FDX_s100
                                 + 0.864051 w_k1_IBM_s100
                                 + 0.877134 w_k1_INTC_s100
                                 + 0.831548 w_k1_KO_s100
                                 + 0.876238 w_k1_MAT_s100
                                 + 0.885363 w_k1_MSFT_s100
                                 + 0.95217 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1583 w_k1_ADM_s100
                                 + 0.839822 w_k1_AXP_s100
                                 + 0.882951 w_k1_DIS_s100
                                 + 1.03457 w_k1_FDX_s100
                                 + 0.683838 w_k1_IBM_s100
                                 + 1.01756 w_k1_INTC_s100
                                 + 0.831663 w_k1_KO_s100
                                 + 0.455733 w_k1_MAT_s100
                                 + 0.932253 w_k1_MSFT_s100
                                 + 0.970015 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.1262 w_k1_ADM_s100
                                 + 1.04403 w_k1_AXP_s100
                                 + 1.86473 w_k1_DIS_s100
                                 + 1.37709 w_k1_FDX_s100
                                 + 1.34825 w_k1_IBM_s100
                                 + 1.37467 w_k1_INTC_s100
                                 + 1.38433 w_k1_KO_s100 + 1.47772 w_k1_MAT_s100
                                 + 1.37949 w_k1_MSFT_s100 + 1.4397 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09099 w_k1_ADM_s100
                                 + 0.921456 w_k1_AXP_s100
                                 + 1.0206 w_k1_DIS_s100
                                 + 0.975687 w_k1_FDX_s100
                                 + 1.03605 w_k1_IBM_s100
                                 + 0.910956 w_k1_INTC_s100
                                 + 0.993455 w_k1_KO_s100
                                 + 1.19566 w_k1_MAT_s100
                                 + 0.978082 w_k1_MSFT_s100
                                 + 0.948247 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.33085 w_k1_ADM_s100
                                 + 1.17645 w_k1_AXP_s100
                                 + 1.27064 w_k1_DIS_s100
                                 + 1.24029 w_k1_FDX_s100
                                 + 1.19409 w_k1_IBM_s100
                                 + 1.04228 w_k1_INTC_s100
                                 + 1.12013 w_k1_KO_s100 + 1.2305 w_k1_MAT_s100
                                 + 1.01239 w_k1_MSFT_s100
                                 + 1.19576 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.11998 w_k1_ADM_s100
                                 + 0.89619 w_k1_AXP_s100
                                 + 0.781628 w_k1_DIS_s100
                                 + 0.802996 w_k1_FDX_s100
                                 + 1.54232 w_k1_IBM_s100
                                 + 0.81024 w_k1_INTC_s100
                                 + 1.13656 w_k1_KO_s100 + 1.57038 w_k1_MAT_s100
                                 + 0.852249 w_k1_MSFT_s100
                                 + 1.09476 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.769429 w_k1_ADM_s100
                                 + 0.700609 w_k1_AXP_s100
                                 + 0.547363 w_k1_DIS_s100
                                 + 0.668622 w_k1_FDX_s100
                                 + 0.948472 w_k1_IBM_s100
                                 + 0.715631 w_k1_INTC_s100
                                 + 0.722034 w_k1_KO_s100
                                 + 0.753026 w_k1_MAT_s100
                                 + 0.624664 w_k1_MSFT_s100
                                 + 0.783523 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.11178 w_k1_ADM_s101
                                 + 1.09133 w_k1_AXP_s101 + 1.4495 w_k1_DIS_s101
                                 + 1.09455 w_k1_FDX_s101
                                 + 1.46417 w_k1_IBM_s101
                                 + 1.18884 w_k1_INTC_s101
                                 + 1.31289 w_k1_KO_s101 + 1.33425 w_k1_MAT_s101
                                 + 1.21834 w_k1_MSFT_s101
                                 + 0.963117 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.872446 w_k1_ADM_s101
                                 + 0.871749 w_k1_AXP_s101
                                 + 0.862386 w_k1_DIS_s101
                                 + 0.851908 w_k1_FDX_s101
                                 + 0.990909 w_k1_IBM_s101
                                 + 0.792114 w_k1_INTC_s101
                                 + 0.858993 w_k1_KO_s101
                                 + 0.750732 w_k1_MAT_s101
                                 + 0.958852 w_k1_MSFT_s101
                                 + 0.901596 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.22146 w_k1_ADM_s101
                                 + 1.36785 w_k1_AXP_s101
                                 + 0.931308 w_k1_DIS_s101
                                 + 0.987373 w_k1_FDX_s101
                                 + 1.23631 w_k1_IBM_s101
                                 + 0.973708 w_k1_INTC_s101
                                 + 1.10238 w_k1_KO_s101 + 1.64594 w_k1_MAT_s101
                                 + 0.895901 w_k1_MSFT_s101
                                 + 1.06008 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.11277 w_k1_ADM_s101
                                 + 0.734083 w_k1_AXP_s101
                                 + 0.669969 w_k1_DIS_s101
                                 + 0.770033 w_k1_FDX_s101
                                 + 0.593362 w_k1_IBM_s101
                                 + 0.797619 w_k1_INTC_s101
                                 + 0.701584 w_k1_KO_s101
                                 + 0.566057 w_k1_MAT_s101
                                 + 0.873511 w_k1_MSFT_s101
                                 + 0.791503 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.16797 w_k1_ADM_s101
                                 + 0.931037 w_k1_AXP_s101
                                 + 1.0608 w_k1_DIS_s101
                                 + 0.989434 w_k1_FDX_s101
                                 + 1.0332 w_k1_IBM_s101
                                 + 0.894904 w_k1_INTC_s101
                                 + 1.08641 w_k1_KO_s101 + 1.60155 w_k1_MAT_s101
                                 + 1.25407 w_k1_MSFT_s101
                                 + 1.06026 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.38996 w_k1_ADM_s101
                                 + 1.28569 w_k1_AXP_s101
                                 + 1.30926 w_k1_DIS_s101
                                 + 1.20885 w_k1_FDX_s101
                                 + 0.879173 w_k1_IBM_s101
                                 + 0.994942 w_k1_INTC_s101
                                 + 1.17713 w_k1_KO_s101 + 1.00124 w_k1_MAT_s101
                                 + 1.2822 w_k1_MSFT_s101 + 1.10173 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.44067 w_k1_ADM_s101
                                 + 1.63231 w_k1_AXP_s101
                                 + 1.72409 w_k1_DIS_s101 + 1.4943 w_k1_FDX_s101
                                 + 1.41851 w_k1_IBM_s101
                                 + 1.21962 w_k1_INTC_s101
                                 + 1.32819 w_k1_KO_s101 + 1.66786 w_k1_MAT_s101
                                 + 1.40663 w_k1_MSFT_s101
                                 + 1.21411 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.02511 w_k1_ADM_s101
                                 + 0.974129 w_k1_AXP_s101
                                 + 1.19812 w_k1_DIS_s101
                                 + 1.14166 w_k1_FDX_s101
                                 + 0.966465 w_k1_IBM_s101
                                 + 0.906573 w_k1_INTC_s101
                                 + 1.07512 w_k1_KO_s101
                                 + 0.885133 w_k1_MAT_s101
                                 + 1.03059 w_k1_MSFT_s101
                                 + 0.945212 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.66089 w_k1_ADM_s102
                                 + 0.828398 w_k1_AXP_s102
                                 + 0.65293 w_k1_DIS_s102
                                 + 0.731346 w_k1_FDX_s102
                                 + 0.782866 w_k1_IBM_s102
                                 + 0.844755 w_k1_INTC_s102
                                 + 0.665764 w_k1_KO_s102
                                 + 0.731164 w_k1_MAT_s102
                                 + 0.769522 w_k1_MSFT_s102
                                 + 0.773229 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.0333 w_k1_ADM_s102
                                 + 1.32423 w_k1_AXP_s102 + 1.1512 w_k1_DIS_s102
                                 + 1.29551 w_k1_FDX_s102
                                 + 0.844427 w_k1_IBM_s102
                                 + 1.4727 w_k1_INTC_s102 + 1.0072 w_k1_KO_s102
                                 + 0.858611 w_k1_MAT_s102
                                 + 1.2136 w_k1_MSFT_s102
                                 + 0.968553 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.02703 w_k1_ADM_s102
                                 + 1.14664 w_k1_AXP_s102
                                 + 1.57056 w_k1_DIS_s102
                                 + 1.14537 w_k1_FDX_s102
                                 + 1.49495 w_k1_IBM_s102
                                 + 1.41194 w_k1_INTC_s102
                                 + 1.35824 w_k1_KO_s102 + 1.67484 w_k1_MAT_s102
                                 + 1.14682 w_k1_MSFT_s102
                                 + 1.11296 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.807106 w_k1_ADM_s102
                                 + 0.926346 w_k1_AXP_s102
                                 + 0.855383 w_k1_DIS_s102
                                 + 1.00688 w_k1_FDX_s102
                                 + 0.692772 w_k1_IBM_s102
                                 + 1.30964 w_k1_INTC_s102
                                 + 0.816992 w_k1_KO_s102
                                 + 0.667273 w_k1_MAT_s102
                                 + 0.978158 w_k1_MSFT_s102
                                 + 0.93979 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.86846 w_k1_ADM_s102
                                 + 1.22894 w_k1_AXP_s102
                                 + 0.899476 w_k1_DIS_s102
                                 + 0.99308 w_k1_FDX_s102
                                 + 0.90699 w_k1_IBM_s102
                                 + 1.44848 w_k1_INTC_s102
                                 + 0.87735 w_k1_KO_s102 + 0.80733 w_k1_MAT_s102
                                 + 0.973768 w_k1_MSFT_s102
                                 + 0.941725 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.54796 w_k1_ADM_s102
                                 + 1.88145 w_k1_AXP_s102 + 1.4576 w_k1_DIS_s102
                                 + 1.56296 w_k1_FDX_s102
                                 + 1.09753 w_k1_IBM_s102
                                 + 1.38722 w_k1_INTC_s102
                                 + 1.11277 w_k1_KO_s102 + 1.46002 w_k1_MAT_s102
                                 + 1.31877 w_k1_MSFT_s102
                                 + 1.21165 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.702735 w_k1_ADM_s102
                                 + 0.932144 w_k1_AXP_s102
                                 + 0.817949 w_k1_DIS_s102
                                 + 0.808696 w_k1_FDX_s102
                                 + 1.14485 w_k1_IBM_s102
                                 + 1.28499 w_k1_INTC_s102
                                 + 0.86222 w_k1_KO_s102
                                 + 0.922931 w_k1_MAT_s102
                                 + 0.850128 w_k1_MSFT_s102
                                 + 0.888115 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.868499 w_k1_ADM_s102
                                 + 1.25576 w_k1_AXP_s102
                                 + 1.14998 w_k1_DIS_s102
                                 + 1.03016 w_k1_FDX_s102
                                 + 1.25181 w_k1_IBM_s102
                                 + 1.05286 w_k1_INTC_s102
                                 + 1.09907 w_k1_KO_s102 + 1.43048 w_k1_MAT_s102
                                 + 1.03817 w_k1_MSFT_s102
                                 + 1.00233 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.8479 w_k1_ADM_s103
                                 + 1.27038 w_k1_AXP_s103
                                 + 0.907867 w_k1_DIS_s103
                                 + 0.946281 w_k1_FDX_s103
                                 + 1.1683 w_k1_IBM_s103
                                 + 0.940744 w_k1_INTC_s103
                                 + 0.941691 w_k1_KO_s103 + 1.3872 w_k1_MAT_s103
                                 + 0.82801 w_k1_MSFT_s103
                                 + 1.09852 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.10982 w_k1_ADM_s103
                                 + 1.04405 w_k1_AXP_s103
                                 + 0.937916 w_k1_DIS_s103
                                 + 1.17376 w_k1_FDX_s103
                                 + 0.802054 w_k1_IBM_s103
                                 + 1.02156 w_k1_INTC_s103
                                 + 0.71751 w_k1_KO_s103
                                 + 0.581258 w_k1_MAT_s103
                                 + 0.896045 w_k1_MSFT_s103
                                 + 1.08295 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.705361 w_k1_ADM_s103
                                 + 0.746436 w_k1_AXP_s103
                                 + 0.677692 w_k1_DIS_s103
                                 + 0.848188 w_k1_FDX_s103
                                 + 0.601675 w_k1_IBM_s103
                                 + 0.973486 w_k1_INTC_s103
                                 + 0.744894 w_k1_KO_s103
                                 + 0.675917 w_k1_MAT_s103
                                 + 0.838046 w_k1_MSFT_s103
                                 + 1.01206 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.09565 w_k1_ADM_s103
                                 + 1.68489 w_k1_AXP_s103
                                 + 1.49056 w_k1_DIS_s103 + 1.4658 w_k1_FDX_s103
                                 + 1.08082 w_k1_IBM_s103
                                 + 1.15429 w_k1_INTC_s103
                                 + 1.11708 w_k1_KO_s103 + 1.54887 w_k1_MAT_s103
                                 + 1.16504 w_k1_MSFT_s103
                                 + 1.40698 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.14654 w_k1_ADM_s103
                                 + 1.07051 w_k1_AXP_s103
                                 + 1.24363 w_k1_DIS_s103
                                 + 1.15829 w_k1_FDX_s103
                                 + 1.04043 w_k1_IBM_s103
                                 + 1.03472 w_k1_INTC_s103
                                 + 1.05969 w_k1_KO_s103 + 1.05833 w_k1_MAT_s103
                                 + 1.09353 w_k1_MSFT_s103
                                 + 1.16255 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.740198 w_k1_ADM_s103
                                 + 0.891641 w_k1_AXP_s103
                                 + 1.15037 w_k1_DIS_s103
                                 + 0.94614 w_k1_FDX_s103
                                 + 1.12049 w_k1_IBM_s103
                                 + 0.936112 w_k1_INTC_s103
                                 + 1.15196 w_k1_KO_s103 + 1.26304 w_k1_MAT_s103
                                 + 0.944033 w_k1_MSFT_s103
                                 + 1.07047 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.663753 w_k1_ADM_s103
                                 + 0.657092 w_k1_AXP_s103
                                 + 0.624831 w_k1_DIS_s103
                                 + 0.642397 w_k1_FDX_s103
                                 + 0.925327 w_k1_IBM_s103
                                 + 0.829832 w_k1_INTC_s103
                                 + 0.812148 w_k1_KO_s103
                                 + 0.843493 w_k1_MAT_s103
                                 + 0.79093 w_k1_MSFT_s103
                                 + 0.882096 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.971394 w_k1_ADM_s103
                                 + 0.7366 w_k1_AXP_s103
                                 + 0.848369 w_k1_DIS_s103
                                 + 0.883872 w_k1_FDX_s103
                                 + 0.849711 w_k1_IBM_s103
                                 + 0.781831 w_k1_INTC_s103
                                 + 0.826455 w_k1_KO_s103
                                 + 0.735595 w_k1_MAT_s103
                                 + 0.918315 w_k1_MSFT_s103
                                 + 1.01272 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.862547 w_k1_ADM_s104
                                 + 0.939162 w_k1_AXP_s104
                                 + 1.06887 w_k1_DIS_s104
                                 + 0.963875 w_k1_FDX_s104
                                 + 0.961995 w_k1_IBM_s104
                                 + 0.960045 w_k1_INTC_s104
                                 + 1.02645 w_k1_KO_s104 + 1.25948 w_k1_MAT_s104
                                 + 0.92635 w_k1_MSFT_s104
                                 + 0.973611 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.13513 w_k1_ADM_s104
                                 + 1.31868 w_k1_AXP_s104
                                 + 1.26171 w_k1_DIS_s104 + 1.0625 w_k1_FDX_s104
                                 + 1.7274 w_k1_IBM_s104
                                 + 1.30123 w_k1_INTC_s104
                                 + 1.44257 w_k1_KO_s104 + 2.32274 w_k1_MAT_s104
                                 + 1.06138 w_k1_MSFT_s104
                                 + 1.19657 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.16839 w_k1_ADM_s104
                                 + 1.12094 w_k1_AXP_s104
                                 + 1.08199 w_k1_DIS_s104
                                 + 1.11952 w_k1_FDX_s104
                                 + 0.768654 w_k1_IBM_s104
                                 + 1.22126 w_k1_INTC_s104
                                 + 0.905429 w_k1_KO_s104
                                 + 1.08511 w_k1_MAT_s104
                                 + 1.14043 w_k1_MSFT_s104 + 1.0781 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.22598 w_k1_ADM_s104
                                 + 1.31026 w_k1_AXP_s104
                                 + 0.981613 w_k1_DIS_s104
                                 + 1.07114 w_k1_FDX_s104
                                 + 1.18056 w_k1_IBM_s104
                                 + 1.08101 w_k1_INTC_s104
                                 + 1.02656 w_k1_KO_s104 + 1.5851 w_k1_MAT_s104
                                 + 0.991332 w_k1_MSFT_s104
                                 + 1.04184 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.851759 w_k1_ADM_s104
                                 + 1.03045 w_k1_AXP_s104
                                 + 0.963174 w_k1_DIS_s104
                                 + 0.869216 w_k1_FDX_s104
                                 + 1.39292 w_k1_IBM_s104
                                 + 1.03974 w_k1_INTC_s104
                                 + 0.974932 w_k1_KO_s104 + 1.7516 w_k1_MAT_s104
                                 + 0.92084 w_k1_MSFT_s104
                                 + 0.965732 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.901983 w_k1_ADM_s104
                                 + 0.724637 w_k1_AXP_s104
                                 + 0.656062 w_k1_DIS_s104
                                 + 0.785675 w_k1_FDX_s104
                                 + 0.586532 w_k1_IBM_s104
                                 + 0.840414 w_k1_INTC_s104
                                 + 0.785608 w_k1_KO_s104
                                 + 0.876338 w_k1_MAT_s104
                                 + 0.823835 w_k1_MSFT_s104
                                 + 0.811408 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.759709 w_k1_ADM_s104
                                 + 0.923523 w_k1_AXP_s104
                                 + 0.668984 w_k1_DIS_s104
                                 + 0.792407 w_k1_FDX_s104
                                 + 0.956825 w_k1_IBM_s104
                                 + 0.962979 w_k1_INTC_s104
                                 + 0.736733 w_k1_KO_s104
                                 + 1.07787 w_k1_MAT_s104
                                 + 0.834501 w_k1_MSFT_s104
                                 + 0.813697 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.56757 w_k1_ADM_s104
                                 + 1.62679 w_k1_AXP_s104
                                 + 1.41869 w_k1_DIS_s104
                                 + 1.48324 w_k1_FDX_s104
                                 + 1.07512 w_k1_IBM_s104
                                 + 1.57421 w_k1_INTC_s104
                                 + 1.15688 w_k1_KO_s104 + 1.2873 w_k1_MAT_s104
                                 + 1.27051 w_k1_MSFT_s104
                                 + 1.22572 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.27987 w_k1_ADM_s105
                                 + 1.36862 w_k1_AXP_s105
                                 + 1.57387 w_k1_DIS_s105
                                 + 1.32272 w_k1_FDX_s105
                                 + 1.21879 w_k1_IBM_s105
                                 + 1.2945 w_k1_INTC_s105 + 1.32038 w_k1_KO_s105
                                 + 1.70103 w_k1_MAT_s105
                                 + 1.33828 w_k1_MSFT_s105
                                 + 0.998816 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.16345 w_k1_ADM_s105
                                 + 1.42597 w_k1_AXP_s105
                                 + 1.28344 w_k1_DIS_s105
                                 + 0.944428 w_k1_FDX_s105
                                 + 1.72299 w_k1_IBM_s105
                                 + 1.00431 w_k1_INTC_s105
                                 + 1.39661 w_k1_KO_s105 + 2.03905 w_k1_MAT_s105
                                 + 0.981703 w_k1_MSFT_s105
                                 + 1.0513 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.871808 w_k1_ADM_s105
                                 + 0.660758 w_k1_AXP_s105
                                 + 0.672605 w_k1_DIS_s105
                                 + 0.615795 w_k1_FDX_s105
                                 + 1.19678 w_k1_IBM_s105
                                 + 0.866686 w_k1_INTC_s105
                                 + 1.09591 w_k1_KO_s105 + 1.00849 w_k1_MAT_s105
                                 + 0.778185 w_k1_MSFT_s105
                                 + 0.806896 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.998585 w_k1_ADM_s105
                                 + 0.666079 w_k1_AXP_s105
                                 + 0.77923 w_k1_DIS_s105
                                 + 0.711843 w_k1_FDX_s105
                                 + 0.884263 w_k1_IBM_s105
                                 + 0.83041 w_k1_INTC_s105
                                 + 0.860796 w_k1_KO_s105
                                 + 0.932258 w_k1_MAT_s105
                                 + 0.922653 w_k1_MSFT_s105
                                 + 0.770615 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.23026 w_k1_ADM_s105
                                 + 1.28414 w_k1_AXP_s105
                                 + 1.15414 w_k1_DIS_s105
                                 + 1.17957 w_k1_FDX_s105
                                 + 0.858503 w_k1_IBM_s105
                                 + 1.2005 w_k1_INTC_s105 + 1.00459 w_k1_KO_s105
                                 + 1.50549 w_k1_MAT_s105
                                 + 1.08315 w_k1_MSFT_s105
                                 + 0.963863 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.21061 w_k1_ADM_s105
                                 + 0.995198 w_k1_AXP_s105
                                 + 1.21603 w_k1_DIS_s105 + 1.1507 w_k1_FDX_s105
                                 + 0.849677 w_k1_IBM_s105
                                 + 1.06733 w_k1_INTC_s105
                                 + 0.972982 w_k1_KO_s105
                                 + 0.840516 w_k1_MAT_s105
                                 + 1.05233 w_k1_MSFT_s105
                                 + 0.905498 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.31082 w_k1_ADM_s105
                                 + 0.88002 w_k1_AXP_s105
                                 + 0.837844 w_k1_DIS_s105
                                 + 1.01426 w_k1_FDX_s105
                                 + 0.690221 w_k1_IBM_s105
                                 + 0.895379 w_k1_INTC_s105
                                 + 0.832882 w_k1_KO_s105
                                 + 0.94745 w_k1_MAT_s105
                                 + 0.948319 w_k1_MSFT_s105
                                 + 0.838661 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.11338 w_k1_ADM_s105
                                 + 0.889348 w_k1_AXP_s105
                                 + 1.00482 w_k1_DIS_s105
                                 + 0.947554 w_k1_FDX_s105
                                 + 1.15363 w_k1_IBM_s105
                                 + 0.916443 w_k1_INTC_s105
                                 + 1.20361 w_k1_KO_s105 + 1.25369 w_k1_MAT_s105
                                 + 1.01031 w_k1_MSFT_s105
                                 + 0.878176 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.852245 w_k1_ADM_s106
                                 + 0.842961 w_k1_AXP_s106
                                 + 1.2472 w_k1_DIS_s106 + 1.07468 w_k1_FDX_s106
                                 + 0.923095 w_k1_IBM_s106
                                 + 1.03872 w_k1_INTC_s106
                                 + 1.06053 w_k1_KO_s106
                                 + 0.770056 w_k1_MAT_s106
                                 + 1.23361 w_k1_MSFT_s106
                                 + 0.95729 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.738925 w_k1_ADM_s106
                                 + 0.733869 w_k1_AXP_s106
                                 + 0.838044 w_k1_DIS_s106
                                 + 0.778761 w_k1_FDX_s106
                                 + 1.12291 w_k1_IBM_s106
                                 + 0.92524 w_k1_INTC_s106
                                 + 0.96747 w_k1_KO_s106
                                 + 0.820543 w_k1_MAT_s106
                                 + 0.880581 w_k1_MSFT_s106
                                 + 0.906985 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.05679 w_k1_ADM_s106
                                 + 1.43045 w_k1_AXP_s106
                                 + 1.87343 w_k1_DIS_s106
                                 + 1.36313 w_k1_FDX_s106 + 1.5774 w_k1_IBM_s106
                                 + 1.44814 w_k1_INTC_s106
                                 + 1.72613 w_k1_KO_s106 + 1.51993 w_k1_MAT_s106
                                 + 1.5666 w_k1_MSFT_s106 + 1.40735 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.07737 w_k1_ADM_s106
                                 + 1.35532 w_k1_AXP_s106
                                 + 1.51909 w_k1_DIS_s106
                                 + 1.29672 w_k1_FDX_s106
                                 + 1.09996 w_k1_IBM_s106
                                 + 1.06783 w_k1_INTC_s106
                                 + 1.34361 w_k1_KO_s106 + 1.41886 w_k1_MAT_s106
                                 + 1.23384 w_k1_MSFT_s106
                                 + 1.22784 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.823914 w_k1_ADM_s106
                                 + 1.02219 w_k1_AXP_s106
                                 + 1.07371 w_k1_DIS_s106
                                 + 0.944118 w_k1_FDX_s106
                                 + 1.31909 w_k1_IBM_s106
                                 + 1.02342 w_k1_INTC_s106
                                 + 1.20019 w_k1_KO_s106 + 1.20937 w_k1_MAT_s106
                                 + 1.05068 w_k1_MSFT_s106
                                 + 1.00535 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.826289 w_k1_ADM_s106
                                 + 0.872087 w_k1_AXP_s106
                                 + 0.892347 w_k1_DIS_s106
                                 + 0.921156 w_k1_FDX_s106
                                 + 0.792916 w_k1_IBM_s106
                                 + 1.17192 w_k1_INTC_s106
                                 + 0.936698 w_k1_KO_s106
                                 + 0.770814 w_k1_MAT_s106
                                 + 1.14141 w_k1_MSFT_s106
                                 + 0.905487 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.967417 w_k1_ADM_s106
                                 + 1.2979 w_k1_AXP_s106 + 1.2605 w_k1_DIS_s106
                                 + 1.19154 w_k1_FDX_s106
                                 + 0.802197 w_k1_IBM_s106
                                 + 1.12356 w_k1_INTC_s106
                                 + 0.989158 w_k1_KO_s106
                                 + 0.996103 w_k1_MAT_s106
                                 + 1.24403 w_k1_MSFT_s106
                                 + 0.926876 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.677167 w_k1_ADM_s106
                                 + 0.729963 w_k1_AXP_s106
                                 + 0.618295 w_k1_DIS_s106
                                 + 0.718373 w_k1_FDX_s106
                                 + 0.847654 w_k1_IBM_s106
                                 + 0.829829 w_k1_INTC_s106
                                 + 0.714448 w_k1_KO_s106
                                 + 0.731768 w_k1_MAT_s106
                                 + 0.823352 w_k1_MSFT_s106
                                 + 0.860603 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.791522 w_k1_ADM_s107
                                 + 0.990262 w_k1_AXP_s107
                                 + 0.800675 w_k1_DIS_s107
                                 + 0.872194 w_k1_FDX_s107
                                 + 1.01519 w_k1_IBM_s107
                                 + 0.76222 w_k1_INTC_s107
                                 + 0.853902 w_k1_KO_s107
                                 + 1.26071 w_k1_MAT_s107
                                 + 0.825021 w_k1_MSFT_s107
                                 + 0.913614 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.10178 w_k1_ADM_s107
                                 + 1.4277 w_k1_AXP_s107
                                 + 0.991983 w_k1_DIS_s107
                                 + 1.04229 w_k1_FDX_s107
                                 + 1.14519 w_k1_IBM_s107
                                 + 1.09219 w_k1_INTC_s107
                                 + 1.05591 w_k1_KO_s107 + 1.31116 w_k1_MAT_s107
                                 + 1.03626 w_k1_MSFT_s107
                                 + 1.12043 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.966717 w_k1_ADM_s107
                                 + 1.18779 w_k1_AXP_s107 + 1.5428 w_k1_DIS_s107
                                 + 1.20966 w_k1_FDX_s107
                                 + 1.53637 w_k1_IBM_s107
                                 + 1.06153 w_k1_INTC_s107
                                 + 1.40358 w_k1_KO_s107 + 2.23679 w_k1_MAT_s107
                                 + 1.22604 w_k1_MSFT_s107
                                 + 1.14368 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.945693 w_k1_ADM_s107
                                 + 0.7661 w_k1_AXP_s107
                                 + 0.791995 w_k1_DIS_s107
                                 + 0.756096 w_k1_FDX_s107
                                 + 0.755227 w_k1_IBM_s107
                                 + 0.861075 w_k1_INTC_s107
                                 + 0.767057 w_k1_KO_s107
                                 + 0.892779 w_k1_MAT_s107
                                 + 0.873551 w_k1_MSFT_s107
                                 + 0.878186 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.10118 w_k1_ADM_s107
                                 + 1.08518 w_k1_AXP_s107
                                 + 1.07059 w_k1_DIS_s107
                                 + 1.22398 w_k1_FDX_s107
                                 + 0.694821 w_k1_IBM_s107
                                 + 1.04438 w_k1_INTC_s107
                                 + 0.812719 w_k1_KO_s107
                                 + 0.666611 w_k1_MAT_s107
                                 + 1.05748 w_k1_MSFT_s107
                                 + 0.947779 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.03028 w_k1_ADM_s107
                                 + 1.21245 w_k1_AXP_s107
                                 + 1.26386 w_k1_DIS_s107
                                 + 1.18601 w_k1_FDX_s107
                                 + 1.01524 w_k1_IBM_s107
                                 + 1.12035 w_k1_INTC_s107
                                 + 1.11961 w_k1_KO_s107 + 1.19061 w_k1_MAT_s107
                                 + 1.06855 w_k1_MSFT_s107
                                 + 1.03533 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.792521 w_k1_ADM_s107
                                 + 0.723006 w_k1_AXP_s107
                                 + 0.559758 w_k1_DIS_s107
                                 + 0.7631 w_k1_FDX_s107
                                 + 0.588378 w_k1_IBM_s107
                                 + 0.871788 w_k1_INTC_s107
                                 + 0.531031 w_k1_KO_s107
                                 + 0.452885 w_k1_MAT_s107
                                 + 0.774205 w_k1_MSFT_s107
                                 + 0.759068 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.86245 w_k1_ADM_s107
                                 + 0.880353 w_k1_AXP_s107
                                 + 0.975591 w_k1_DIS_s107
                                 + 0.956125 w_k1_FDX_s107
                                 + 0.870758 w_k1_IBM_s107
                                 + 1.00688 w_k1_INTC_s107
                                 + 0.903334 w_k1_KO_s107
                                 + 0.901389 w_k1_MAT_s107
                                 + 0.895858 w_k1_MSFT_s107
                                 + 0.848314 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.728289 w_k1_ADM_s110
                                 + 0.688739 w_k1_AXP_s110
                                 + 0.692657 w_k1_DIS_s110
                                 + 0.705085 w_k1_FDX_s110
                                 + 0.761945 w_k1_IBM_s110
                                 + 0.91061 w_k1_INTC_s110
                                 + 0.654953 w_k1_KO_s110
                                 + 0.548235 w_k1_MAT_s110
                                 + 0.733847 w_k1_MSFT_s110
                                 + 0.72141 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.03228 w_k1_ADM_s110
                                 + 1.37133 w_k1_AXP_s110
                                 + 1.18089 w_k1_DIS_s110
                                 + 1.09079 w_k1_FDX_s110
                                 + 1.22463 w_k1_IBM_s110
                                 + 1.04515 w_k1_INTC_s110
                                 + 1.04772 w_k1_KO_s110 + 1.31419 w_k1_MAT_s110
                                 + 0.9542 w_k1_MSFT_s110 + 1.15273 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.11339 w_k1_ADM_s110
                                 + 1.24677 w_k1_AXP_s110 + 1.4562 w_k1_DIS_s110
                                 + 1.25811 w_k1_FDX_s110
                                 + 0.803175 w_k1_IBM_s110
                                 + 1.20667 w_k1_INTC_s110
                                 + 0.882829 w_k1_KO_s110
                                 + 0.710964 w_k1_MAT_s110
                                 + 1.12922 w_k1_MSFT_s110 + 0.9823 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.01263 w_k1_ADM_s110
                                 + 0.901641 w_k1_AXP_s110
                                 + 0.949881 w_k1_DIS_s110
                                 + 0.932151 w_k1_FDX_s110
                                 + 0.899592 w_k1_IBM_s110
                                 + 1.17484 w_k1_INTC_s110
                                 + 0.921125 w_k1_KO_s110
                                 + 0.736716 w_k1_MAT_s110
                                 + 1.03184 w_k1_MSFT_s110
                                 + 0.974243 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.812595 w_k1_ADM_s110
                                 + 0.840724 w_k1_AXP_s110
                                 + 0.937247 w_k1_DIS_s110
                                 + 0.904487 w_k1_FDX_s110
                                 + 1.1366 w_k1_IBM_s110
                                 + 1.00486 w_k1_INTC_s110
                                 + 0.93828 w_k1_KO_s110 + 1.06149 w_k1_MAT_s110
                                 + 0.917693 w_k1_MSFT_s110
                                 + 0.964697 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.734438 w_k1_ADM_s110
                                 + 0.881433 w_k1_AXP_s110
                                 + 0.758127 w_k1_DIS_s110
                                 + 0.820181 w_k1_FDX_s110
                                 + 1.36922 w_k1_IBM_s110
                                 + 0.928631 w_k1_INTC_s110
                                 + 1.10652 w_k1_KO_s110 + 1.26399 w_k1_MAT_s110
                                 + 0.76843 w_k1_MSFT_s110
                                 + 1.03834 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.804364 w_k1_ADM_s110
                                 + 1.10803 w_k1_AXP_s110
                                 + 0.746772 w_k1_DIS_s110
                                 + 0.8648 w_k1_FDX_s110
                                 + 0.908509 w_k1_IBM_s110
                                 + 0.954949 w_k1_INTC_s110
                                 + 0.838705 w_k1_KO_s110
                                 + 0.708475 w_k1_MAT_s110
                                 + 0.766678 w_k1_MSFT_s110
                                 + 0.998812 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.948822 w_k1_ADM_s110
                                 + 1.19225 w_k1_AXP_s110
                                 + 1.58672 w_k1_DIS_s110
                                 + 1.24765 w_k1_FDX_s110
                                 + 1.46877 w_k1_IBM_s110
                                 + 1.53361 w_k1_INTC_s110
                                 + 1.30014 w_k1_KO_s110
                                 + 0.942209 w_k1_MAT_s110
                                 + 1.19552 w_k1_MSFT_s110 + 1.2427 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.794482 w_k1_ADM_s111
                                 + 0.691159 w_k1_AXP_s111
                                 + 0.708553 w_k1_DIS_s111
                                 + 0.753682 w_k1_FDX_s111
                                 + 0.711331 w_k1_IBM_s111
                                 + 0.824409 w_k1_INTC_s111
                                 + 0.860475 w_k1_KO_s111
                                 + 0.828987 w_k1_MAT_s111
                                 + 0.732381 w_k1_MSFT_s111
                                 + 0.919405 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.15595 w_k1_ADM_s111
                                 + 0.914781 w_k1_AXP_s111
                                 + 1.09185 w_k1_DIS_s111
                                 + 1.05895 w_k1_FDX_s111
                                 + 0.700363 w_k1_IBM_s111
                                 + 0.832982 w_k1_INTC_s111
                                 + 0.865596 w_k1_KO_s111
                                 + 0.929592 w_k1_MAT_s111
                                 + 0.943075 w_k1_MSFT_s111
                                 + 1.00817 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.03248 w_k1_ADM_s111
                                 + 1.27547 w_k1_AXP_s111
                                 + 1.07836 w_k1_DIS_s111
                                 + 0.93665 w_k1_FDX_s111
                                 + 1.55953 w_k1_IBM_s111
                                 + 0.97727 w_k1_INTC_s111
                                 + 1.28547 w_k1_KO_s111 + 2.00019 w_k1_MAT_s111
                                 + 0.881435 w_k1_MSFT_s111
                                 + 1.12103 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.994724 w_k1_ADM_s111
                                 + 1.02489 w_k1_AXP_s111
                                 + 0.94141 w_k1_DIS_s111
                                 + 0.999292 w_k1_FDX_s111
                                 + 1.10887 w_k1_IBM_s111
                                 + 0.90541 w_k1_INTC_s111
                                 + 1.16993 w_k1_KO_s111 + 1.39475 w_k1_MAT_s111
                                 + 0.984748 w_k1_MSFT_s111
                                 + 1.11923 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.880988 w_k1_ADM_s111
                                 + 0.969967 w_k1_AXP_s111
                                 + 0.717241 w_k1_DIS_s111
                                 + 0.827529 w_k1_FDX_s111
                                 + 0.981861 w_k1_IBM_s111
                                 + 0.782325 w_k1_INTC_s111
                                 + 0.797842 w_k1_KO_s111
                                 + 0.881979 w_k1_MAT_s111
                                 + 0.788294 w_k1_MSFT_s111
                                 + 0.913396 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.06696 w_k1_ADM_s111
                                 + 1.08459 w_k1_AXP_s111
                                 + 1.26451 w_k1_DIS_s111
                                 + 1.31853 w_k1_FDX_s111
                                 + 0.984333 w_k1_IBM_s111
                                 + 0.917045 w_k1_INTC_s111
                                 + 1.0353 w_k1_KO_s111 + 1.24171 w_k1_MAT_s111
                                 + 1.20101 w_k1_MSFT_s111
                                 + 1.15094 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.708394 w_k1_ADM_s111
                                 + 0.650319 w_k1_AXP_s111
                                 + 0.539248 w_k1_DIS_s111
                                 + 0.61219 w_k1_FDX_s111
                                 + 0.96042 w_k1_IBM_s111
                                 + 0.661187 w_k1_INTC_s111
                                 + 0.759564 w_k1_KO_s111
                                 + 0.983606 w_k1_MAT_s111
                                 + 0.664964 w_k1_MSFT_s111
                                 + 0.761527 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.35879 w_k1_ADM_s111
                                 + 1.59594 w_k1_AXP_s111
                                 + 1.73343 w_k1_DIS_s111
                                 + 1.45799 w_k1_FDX_s111
                                 + 1.24517 w_k1_IBM_s111
                                 + 0.926084 w_k1_INTC_s111
                                 + 1.32331 w_k1_KO_s111 + 1.8058 w_k1_MAT_s111
                                 + 1.1307 w_k1_MSFT_s111 + 1.35116 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.885513 w_k1_ADM_s112
                                 + 1.03371 w_k1_AXP_s112
                                 + 1.02952 w_k1_DIS_s112
                                 + 0.933104 w_k1_FDX_s112
                                 + 1.3528 w_k1_IBM_s112
                                 + 1.00235 w_k1_INTC_s112 + 1.1145 w_k1_KO_s112
                                 + 1.33473 w_k1_MAT_s112
                                 + 0.971259 w_k1_MSFT_s112
                                 + 1.12638 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.859024 w_k1_ADM_s112
                                 + 0.958857 w_k1_AXP_s112
                                 + 0.893258 w_k1_DIS_s112
                                 + 1.05603 w_k1_FDX_s112
                                 + 0.91514 w_k1_IBM_s112
                                 + 0.912908 w_k1_INTC_s112
                                 + 0.927758 w_k1_KO_s112 + 1.332 w_k1_MAT_s112
                                 + 0.973389 w_k1_MSFT_s112
                                 + 1.02167 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.840366 w_k1_ADM_s112
                                 + 0.677768 w_k1_AXP_s112
                                 + 0.894098 w_k1_DIS_s112
                                 + 0.839648 w_k1_FDX_s112
                                 + 1.05844 w_k1_IBM_s112
                                 + 0.900164 w_k1_INTC_s112
                                 + 1.02059 w_k1_KO_s112 + 0.78373 w_k1_MAT_s112
                                 + 0.93039 w_k1_MSFT_s112
                                 + 1.05789 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.987564 w_k1_ADM_s112
                                 + 1.32028 w_k1_AXP_s112
                                 + 1.24586 w_k1_DIS_s112
                                 + 1.33783 w_k1_FDX_s112
                                 + 0.944984 w_k1_IBM_s112
                                 + 1.26705 w_k1_INTC_s112
                                 + 0.978151 w_k1_KO_s112
                                 + 1.16391 w_k1_MAT_s112
                                 + 1.2076 w_k1_MSFT_s112 + 1.1733 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.743969 w_k1_ADM_s112
                                 + 0.633646 w_k1_AXP_s112
                                 + 0.64403 w_k1_DIS_s112
                                 + 0.734748 w_k1_FDX_s112
                                 + 0.820911 w_k1_IBM_s112
                                 + 0.697078 w_k1_INTC_s112
                                 + 0.798921 w_k1_KO_s112
                                 + 0.856933 w_k1_MAT_s112
                                 + 0.723531 w_k1_MSFT_s112
                                 + 0.846753 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.897254 w_k1_ADM_s112
                                 + 0.722479 w_k1_AXP_s112
                                 + 0.85137 w_k1_DIS_s112
                                 + 0.985398 w_k1_FDX_s112
                                 + 0.67152 w_k1_IBM_s112
                                 + 0.84743 w_k1_INTC_s112
                                 + 0.845464 w_k1_KO_s112
                                 + 0.745767 w_k1_MAT_s112
                                 + 0.948706 w_k1_MSFT_s112
                                 + 0.972257 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.23637 w_k1_ADM_s112
                                 + 1.36348 w_k1_AXP_s112
                                 + 1.85253 w_k1_DIS_s112
                                 + 1.34325 w_k1_FDX_s112
                                 + 1.45387 w_k1_IBM_s112
                                 + 1.18745 w_k1_INTC_s112
                                 + 1.39086 w_k1_KO_s112 + 1.33423 w_k1_MAT_s112
                                 + 1.3522 w_k1_MSFT_s112 + 1.27827 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.20776 w_k1_ADM_s112
                                 + 1.36449 w_k1_AXP_s112
                                 + 1.12665 w_k1_DIS_s112
                                 + 1.07152 w_k1_FDX_s112
                                 + 1.35348 w_k1_IBM_s112
                                 + 0.933603 w_k1_INTC_s112
                                 + 1.18464 w_k1_KO_s112 + 1.7641 w_k1_MAT_s112
                                 + 1.00708 w_k1_MSFT_s112
                                 + 1.20976 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.923575 w_k1_ADM_s113
                                 + 0.998958 w_k1_AXP_s113
                                 + 1.16827 w_k1_DIS_s113
                                 + 1.00272 w_k1_FDX_s113
                                 + 0.87131 w_k1_IBM_s113
                                 + 1.10398 w_k1_INTC_s113
                                 + 0.990198 w_k1_KO_s113
                                 + 0.814847 w_k1_MAT_s113
                                 + 1.03945 w_k1_MSFT_s113
                                 + 1.08634 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.862639 w_k1_ADM_s113
                                 + 0.652634 w_k1_AXP_s113
                                 + 0.596739 w_k1_DIS_s113
                                 + 0.694483 w_k1_FDX_s113
                                 + 0.67135 w_k1_IBM_s113
                                 + 0.751721 w_k1_INTC_s113
                                 + 0.619718 w_k1_KO_s113
                                 + 0.448405 w_k1_MAT_s113
                                 + 0.630406 w_k1_MSFT_s113
                                 + 0.746039 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.10341 w_k1_ADM_s113
                                 + 1.5705 w_k1_AXP_s113 + 1.66663 w_k1_DIS_s113
                                 + 1.37742 w_k1_FDX_s113
                                 + 1.05655 w_k1_IBM_s113
                                 + 1.33678 w_k1_INTC_s113
                                 + 1.06805 w_k1_KO_s113 + 1.25311 w_k1_MAT_s113
                                 + 1.13476 w_k1_MSFT_s113
                                 + 1.11623 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.14675 w_k1_ADM_s113
                                 + 1.13331 w_k1_AXP_s113
                                 + 1.47512 w_k1_DIS_s113
                                 + 1.11762 w_k1_FDX_s113
                                 + 1.51458 w_k1_IBM_s113
                                 + 1.27212 w_k1_INTC_s113
                                 + 1.32276 w_k1_KO_s113 + 1.67095 w_k1_MAT_s113
                                 + 1.21531 w_k1_MSFT_s113
                                 + 1.16431 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.734052 w_k1_ADM_s113
                                 + 0.786749 w_k1_AXP_s113
                                 + 0.582772 w_k1_DIS_s113
                                 + 0.693235 w_k1_FDX_s113
                                 + 0.936687 w_k1_IBM_s113
                                 + 0.853496 w_k1_INTC_s113
                                 + 0.704268 w_k1_KO_s113
                                 + 0.850052 w_k1_MAT_s113
                                 + 0.732154 w_k1_MSFT_s113
                                 + 0.838881 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.987512 w_k1_ADM_s113
                                 + 0.957097 w_k1_AXP_s113
                                 + 0.847762 w_k1_DIS_s113
                                 + 0.951514 w_k1_FDX_s113
                                 + 0.801098 w_k1_IBM_s113
                                 + 0.988621 w_k1_INTC_s113
                                 + 0.711532 w_k1_KO_s113
                                 + 0.570599 w_k1_MAT_s113
                                 + 0.840171 w_k1_MSFT_s113
                                 + 0.934784 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.951215 w_k1_ADM_s113
                                 + 1.09012 w_k1_AXP_s113
                                 + 0.856006 w_k1_DIS_s113
                                 + 0.837643 w_k1_FDX_s113
                                 + 1.19261 w_k1_IBM_s113
                                 + 1.05803 w_k1_INTC_s113
                                 + 0.990633 w_k1_KO_s113
                                 + 1.32274 w_k1_MAT_s113
                                 + 0.919339 w_k1_MSFT_s113
                                 + 0.974813 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.35839 w_k1_ADM_s113
                                 + 1.13943 w_k1_AXP_s113
                                 + 1.08309 w_k1_DIS_s113
                                 + 1.13319 w_k1_FDX_s113 + 1.0613 w_k1_IBM_s113
                                 + 1.02401 w_k1_INTC_s113
                                 + 0.949369 w_k1_KO_s113 + 1.1018 w_k1_MAT_s113
                                 + 0.957825 w_k1_MSFT_s113
                                 + 1.06243 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.26533 w_k1_ADM_s114
                                 + 0.991275 w_k1_AXP_s114
                                 + 0.773484 w_k1_DIS_s114
                                 + 0.772392 w_k1_FDX_s114
                                 + 1.15671 w_k1_IBM_s114
                                 + 0.778552 w_k1_INTC_s114
                                 + 0.949871 w_k1_KO_s114
                                 + 0.998699 w_k1_MAT_s114
                                 + 0.824109 w_k1_MSFT_s114
                                 + 0.934889 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.44396 w_k1_ADM_s114
                                 + 1.829 w_k1_AXP_s114 + 1.87817 w_k1_DIS_s114
                                 + 1.41567 w_k1_FDX_s114
                                 + 1.32185 w_k1_IBM_s114
                                 + 1.43202 w_k1_INTC_s114
                                 + 1.37288 w_k1_KO_s114 + 1.45191 w_k1_MAT_s114
                                 + 1.25093 w_k1_MSFT_s114
                                 + 1.21575 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.3467 w_k1_ADM_s114
                                 + 1.36476 w_k1_AXP_s114
                                 + 1.01385 w_k1_DIS_s114
                                 + 0.928553 w_k1_FDX_s114
                                 + 1.64452 w_k1_IBM_s114
                                 + 1.0038 w_k1_INTC_s114 + 1.3959 w_k1_KO_s114
                                 + 1.58101 w_k1_MAT_s114
                                 + 0.916944 w_k1_MSFT_s114
                                 + 1.17807 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.07501 w_k1_ADM_s114
                                 + 1.0382 w_k1_AXP_s114
                                 + 0.669534 w_k1_DIS_s114
                                 + 0.792383 w_k1_FDX_s114
                                 + 0.919379 w_k1_IBM_s114
                                 + 0.714488 w_k1_INTC_s114
                                 + 0.772132 w_k1_KO_s114 + 1.2756 w_k1_MAT_s114
                                 + 0.690057 w_k1_MSFT_s114
                                 + 0.891386 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.49094 w_k1_ADM_s114
                                 + 1.1766 w_k1_AXP_s114 + 1.36227 w_k1_DIS_s114
                                 + 1.12328 w_k1_FDX_s114
                                 + 1.17323 w_k1_IBM_s114
                                 + 1.00995 w_k1_INTC_s114
                                 + 1.25745 w_k1_KO_s114 + 1.18778 w_k1_MAT_s114
                                 + 1.06447 w_k1_MSFT_s114
                                 + 1.29313 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.941177 w_k1_ADM_s114
                                 + 1.31654 w_k1_AXP_s114
                                 + 1.25817 w_k1_DIS_s114
                                 + 1.13121 w_k1_FDX_s114
                                 + 0.967061 w_k1_IBM_s114
                                 + 1.16291 w_k1_INTC_s114
                                 + 0.999344 w_k1_KO_s114
                                 + 1.10521 w_k1_MAT_s114
                                 + 1.00805 w_k1_MSFT_s114
                                 + 0.992122 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.869103 w_k1_ADM_s114
                                 + 0.693439 w_k1_AXP_s114
                                 + 0.69025 w_k1_DIS_s114
                                 + 0.814694 w_k1_FDX_s114
                                 + 0.737173 w_k1_IBM_s114
                                 + 0.99304 w_k1_INTC_s114
                                 + 0.844691 w_k1_KO_s114
                                 + 0.64059 w_k1_MAT_s114
                                 + 1.00262 w_k1_MSFT_s114
                                 + 0.934693 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.17239 w_k1_ADM_s114
                                 + 1.04281 w_k1_AXP_s114 + 1.1645 w_k1_DIS_s114
                                 + 1.13491 w_k1_FDX_s114
                                 + 0.842614 w_k1_IBM_s114
                                 + 1.01939 w_k1_INTC_s114
                                 + 0.981556 w_k1_KO_s114
                                 + 0.868446 w_k1_MAT_s114
                                 + 1.00779 w_k1_MSFT_s114
                                 + 1.14483 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.2105 w_k1_ADM_s115
                                 + 1.123 w_k1_AXP_s115 + 1.14391 w_k1_DIS_s115
                                 + 1.14992 w_k1_FDX_s115
                                 + 0.992774 w_k1_IBM_s115
                                 + 1.11361 w_k1_INTC_s115
                                 + 1.12018 w_k1_KO_s115 + 1.00275 w_k1_MAT_s115
                                 + 1.24227 w_k1_MSFT_s115
                                 + 1.01386 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.899535 w_k1_ADM_s115
                                 + 1.0188 w_k1_AXP_s115 + 0.95193 w_k1_DIS_s115
                                 + 0.890506 w_k1_FDX_s115
                                 + 1.46019 w_k1_IBM_s115
                                 + 1.00524 w_k1_INTC_s115
                                 + 1.18308 w_k1_KO_s115 + 1.82295 w_k1_MAT_s115
                                 + 1.05247 w_k1_MSFT_s115
                                 + 0.973072 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.934031 w_k1_ADM_s115
                                 + 0.894794 w_k1_AXP_s115
                                 + 0.956581 w_k1_DIS_s115
                                 + 1.02187 w_k1_FDX_s115
                                 + 0.860067 w_k1_IBM_s115
                                 + 1.00376 w_k1_INTC_s115
                                 + 0.932223 w_k1_KO_s115
                                 + 0.762376 w_k1_MAT_s115
                                 + 1.12303 w_k1_MSFT_s115 + 0.8482 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.19073 w_k1_ADM_s115
                                 + 1.09823 w_k1_AXP_s115
                                 + 0.895843 w_k1_DIS_s115
                                 + 0.933136 w_k1_FDX_s115
                                 + 1.24396 w_k1_IBM_s115
                                 + 1.03099 w_k1_INTC_s115
                                 + 1.15049 w_k1_KO_s115 + 1.26274 w_k1_MAT_s115
                                 + 1.01326 w_k1_MSFT_s115
                                 + 0.89894 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.40225 w_k1_ADM_s115
                                 + 1.30193 w_k1_AXP_s115
                                 + 1.71162 w_k1_DIS_s115 + 1.3745 w_k1_FDX_s115
                                 + 1.1585 w_k1_IBM_s115
                                 + 1.38534 w_k1_INTC_s115
                                 + 1.60195 w_k1_KO_s115 + 1.31353 w_k1_MAT_s115
                                 + 1.47734 w_k1_MSFT_s115
                                 + 1.13569 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.42072 w_k1_ADM_s115
                                 + 1.36075 w_k1_AXP_s115
                                 + 1.34875 w_k1_DIS_s115
                                 + 1.23935 w_k1_FDX_s115 + 1.5517 w_k1_IBM_s115
                                 + 1.1074 w_k1_INTC_s115 + 1.54071 w_k1_KO_s115
                                 + 1.88235 w_k1_MAT_s115
                                 + 1.50256 w_k1_MSFT_s115
                                 + 1.16038 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.08034 w_k1_ADM_s115
                                 + 0.833848 w_k1_AXP_s115
                                 + 0.627594 w_k1_DIS_s115
                                 + 0.751501 w_k1_FDX_s115
                                 + 0.951259 w_k1_IBM_s115
                                 + 0.631733 w_k1_INTC_s115
                                 + 0.953881 w_k1_KO_s115
                                 + 0.745551 w_k1_MAT_s115
                                 + 0.780548 w_k1_MSFT_s115
                                 + 0.769245 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.13583 w_k1_ADM_s115
                                 + 0.904567 w_k1_AXP_s115
                                 + 0.754878 w_k1_DIS_s115
                                 + 0.936771 w_k1_FDX_s115
                                 + 0.651791 w_k1_IBM_s115
                                 + 0.900723 w_k1_INTC_s115
                                 + 0.751203 w_k1_KO_s115
                                 + 0.65509 w_k1_MAT_s115
                                 + 0.988024 w_k1_MSFT_s115
                                 + 0.706081 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.20469 w_k1_ADM_s116
                                 + 1.24678 w_k1_AXP_s116
                                 + 1.51784 w_k1_DIS_s116
                                 + 1.32568 w_k1_FDX_s116
                                 + 0.97302 w_k1_IBM_s116
                                 + 1.42264 w_k1_INTC_s116 + 1.0958 w_k1_KO_s116
                                 + 0.805968 w_k1_MAT_s116
                                 + 1.24555 w_k1_MSFT_s116
                                 + 1.19521 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.12341 w_k1_ADM_s116
                                 + 1.5945 w_k1_AXP_s116 + 1.70295 w_k1_DIS_s116
                                 + 1.36151 w_k1_FDX_s116
                                 + 1.38605 w_k1_IBM_s116 + 1.356 w_k1_INTC_s116
                                 + 1.4118 w_k1_KO_s116 + 1.51778 w_k1_MAT_s116
                                 + 1.32054 w_k1_MSFT_s116
                                 + 1.29213 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.850571 w_k1_ADM_s116
                                 + 0.577222 w_k1_AXP_s116
                                 + 0.538591 w_k1_DIS_s116
                                 + 0.600527 w_k1_FDX_s116
                                 + 0.825986 w_k1_IBM_s116
                                 + 0.714681 w_k1_INTC_s116
                                 + 0.64825 w_k1_KO_s116
                                 + 0.623896 w_k1_MAT_s116
                                 + 0.679408 w_k1_MSFT_s116
                                 + 0.736316 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.878383 w_k1_ADM_s116
                                 + 0.775226 w_k1_AXP_s116
                                 + 0.982411 w_k1_DIS_s116
                                 + 1.00083 w_k1_FDX_s116
                                 + 0.875637 w_k1_IBM_s116
                                 + 0.915573 w_k1_INTC_s116
                                 + 0.837465 w_k1_KO_s116
                                 + 0.647719 w_k1_MAT_s116
                                 + 1.0471 w_k1_MSFT_s116
                                 + 0.973102 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.14295 w_k1_ADM_s116
                                 + 1.16483 w_k1_AXP_s116
                                 + 1.13583 w_k1_DIS_s116
                                 + 1.14838 w_k1_FDX_s116
                                 + 1.16988 w_k1_IBM_s116
                                 + 0.927645 w_k1_INTC_s116
                                 + 0.950151 w_k1_KO_s116
                                 + 1.05225 w_k1_MAT_s116
                                 + 1.05083 w_k1_MSFT_s116
                                 + 1.08235 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.7796 w_k1_ADM_s116
                                 + 1.48285 w_k1_AXP_s116
                                 + 1.73513 w_k1_DIS_s116
                                 + 1.67262 w_k1_FDX_s116
                                 + 0.959706 w_k1_IBM_s116
                                 + 1.26105 w_k1_INTC_s116
                                 + 1.15619 w_k1_KO_s116
                                 + 0.842287 w_k1_MAT_s116
                                 + 1.30182 w_k1_MSFT_s116
                                 + 1.26767 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.951612 w_k1_ADM_s116
                                 + 0.965888 w_k1_AXP_s116
                                 + 0.709493 w_k1_DIS_s116
                                 + 0.838258 w_k1_FDX_s116
                                 + 0.781439 w_k1_IBM_s116
                                 + 0.813363 w_k1_INTC_s116
                                 + 0.678966 w_k1_KO_s116
                                 + 0.745717 w_k1_MAT_s116
                                 + 0.789522 w_k1_MSFT_s116
                                 + 0.878168 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.976222 w_k1_ADM_s116
                                 + 0.869637 w_k1_AXP_s116
                                 + 0.749844 w_k1_DIS_s116
                                 + 0.853692 w_k1_FDX_s116
                                 + 1.2338 w_k1_IBM_s116
                                 + 0.899921 w_k1_INTC_s116
                                 + 0.930847 w_k1_KO_s116
                                 + 1.18424 w_k1_MAT_s116
                                 + 0.89838 w_k1_MSFT_s116
                                 + 0.945583 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.846036 w_k1_ADM_s117
                                 + 0.861565 w_k1_AXP_s117
                                 + 0.627865 w_k1_DIS_s117
                                 + 0.812618 w_k1_FDX_s117
                                 + 0.770213 w_k1_IBM_s117
                                 + 0.898717 w_k1_INTC_s117
                                 + 0.70468 w_k1_KO_s117
                                 + 0.867579 w_k1_MAT_s117
                                 + 0.776833 w_k1_MSFT_s117
                                 + 0.833157 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.31283 w_k1_ADM_s117
                                 + 1.78425 w_k1_AXP_s117
                                 + 1.45286 w_k1_DIS_s117
                                 + 1.20688 w_k1_FDX_s117
                                 + 1.48765 w_k1_IBM_s117
                                 + 1.15229 w_k1_INTC_s117
                                 + 1.23826 w_k1_KO_s117 + 1.85906 w_k1_MAT_s117
                                 + 1.22513 w_k1_MSFT_s117
                                 + 1.16898 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.925831 w_k1_ADM_s117
                                 + 1.1959 w_k1_AXP_s117 + 0.82639 w_k1_DIS_s117
                                 + 0.85247 w_k1_FDX_s117
                                 + 1.47489 w_k1_IBM_s117
                                 + 0.813609 w_k1_INTC_s117
                                 + 0.851206 w_k1_KO_s117
                                 + 1.93455 w_k1_MAT_s117
                                 + 0.736178 w_k1_MSFT_s117
                                 + 0.970511 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.22112 w_k1_ADM_s117
                                 + 1.02249 w_k1_AXP_s117
                                 + 1.03542 w_k1_DIS_s117
                                 + 0.928432 w_k1_FDX_s117
                                 + 1.13727 w_k1_IBM_s117
                                 + 0.823527 w_k1_INTC_s117
                                 + 1.03366 w_k1_KO_s117 + 1.46802 w_k1_MAT_s117
                                 + 0.900134 w_k1_MSFT_s117
                                 + 1.10552 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.48713 w_k1_ADM_s117
                                 + 1.14822 w_k1_AXP_s117
                                 + 1.42727 w_k1_DIS_s117
                                 + 1.26535 w_k1_FDX_s117
                                 + 0.945919 w_k1_IBM_s117
                                 + 0.999497 w_k1_INTC_s117
                                 + 1.18166 w_k1_KO_s117 + 1.02997 w_k1_MAT_s117
                                 + 1.19047 w_k1_MSFT_s117
                                 + 1.06929 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.09609 w_k1_ADM_s117
                                 + 1.0928 w_k1_AXP_s117 + 0.84524 w_k1_DIS_s117
                                 + 0.91292 w_k1_FDX_s117
                                 + 1.16974 w_k1_IBM_s117
                                 + 0.839341 w_k1_INTC_s117
                                 + 0.98594 w_k1_KO_s117 + 1.94108 w_k1_MAT_s117
                                 + 0.886996 w_k1_MSFT_s117
                                 + 1.01733 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.05685 w_k1_ADM_s117
                                 + 0.851047 w_k1_AXP_s117
                                 + 1.16949 w_k1_DIS_s117
                                 + 1.14089 w_k1_FDX_s117
                                 + 0.96391 w_k1_IBM_s117
                                 + 1.0988 w_k1_INTC_s117 + 1.06601 w_k1_KO_s117
                                 + 1.09385 w_k1_MAT_s117
                                 + 1.12287 w_k1_MSFT_s117 + 1.2583 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.832709 w_k1_ADM_s117
                                 + 0.650772 w_k1_AXP_s117
                                 + 0.751289 w_k1_DIS_s117
                                 + 0.824798 w_k1_FDX_s117
                                 + 0.80551 w_k1_IBM_s117
                                 + 0.915636 w_k1_INTC_s117
                                 + 0.833904 w_k1_KO_s117
                                 + 0.728244 w_k1_MAT_s117
                                 + 0.843822 w_k1_MSFT_s117
                                 + 0.895832 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0195124031270213 w_k1_ADM_s000
                                 + 0.0233569879441698 w_k1_ADM_s001
                                 + 0.0709843609964437 w_k1_ADM_s002
                                 + 0.0261936917774275 w_k1_ADM_s003
                                 + 0.0333878845918856 w_k1_ADM_s004
                                 + 0.0178622159461419 w_k1_ADM_s005
                                 + 0.0339062837721452 w_k1_ADM_s006
                                 + 0.0196615642637993 w_k1_ADM_s007
                                 + 0.0475285883941469 w_k1_ADM_s010
                                 + 0.0295121775619127 w_k1_ADM_s011
                                 + 0.0149984253216502 w_k1_ADM_s012
                                 + 0.0285470278682028 w_k1_ADM_s013
                                 + 0.0354264080401395 w_k1_ADM_s014
                                 + 0.0517662326074154 w_k1_ADM_s015
                                 + 0.0152401501500486 w_k1_ADM_s016
                                 + 0.0318987520431767 w_k1_ADM_s017
                                 + 0.0420926590835146 w_k1_ADM_s100
                                 + 0.0411891088625718 w_k1_ADM_s101
                                 + 0.0147737945467986 w_k1_ADM_s102
                                 + 0.019458108770605 w_k1_ADM_s103
                                 + 0.0624639851502296 w_k1_ADM_s104
                                 + 0.0229370122613388 w_k1_ADM_s105
                                 + 0.050303947861991 w_k1_ADM_s106
                                 + 0.0241491490503154 w_k1_ADM_s107
                                 + 0.0183269997402847 w_k1_ADM_s110
                                 + 0.0213136254543024 w_k1_ADM_s111
                                 + 0.0286313023121704 w_k1_ADM_s112
                                 + 0.0503437107255508 w_k1_ADM_s113
                                 + 0.0154011609917633 w_k1_ADM_s114
                                 + 0.0327700567757628 w_k1_ADM_s115
                                 + 0.0386621296064657 w_k1_ADM_s116
                                 + 0.0206896521718822 w_k1_ADM_s117
                                 + 0.0234660891939491 w_k1_AXP_s000
                                 + 0.023586967093631 w_k1_AXP_s001
                                 + 0.0730716848303834 w_k1_AXP_s002
                                 + 0.0233524504446688 w_k1_AXP_s003
                                 + 0.0383398865005135 w_k1_AXP_s004
                                 + 0.0172217110280032 w_k1_AXP_s005
                                 + 0.0313136513099428 w_k1_AXP_s006
                                 + 0.0211277271969179 w_k1_AXP_s007
                                 + 0.0519871611802655 w_k1_AXP_s010
                                 + 0.031374524504084 w_k1_AXP_s011
                                 + 0.016452069699092 w_k1_AXP_s012
                                 + 0.0257815192044791 w_k1_AXP_s013
                                 + 0.0383466628368142 w_k1_AXP_s014
                                 + 0.048338933722063 w_k1_AXP_s015
                                 + 0.0166487640566447 w_k1_AXP_s016
                                 + 0.0345554023536877 w_k1_AXP_s017
                                 + 0.0355377519897931 w_k1_AXP_s100
                                 + 0.0384805457929289 w_k1_AXP_s101
                                 + 0.0189834230174257 w_k1_AXP_s102
                                 + 0.0224668841687253 w_k1_AXP_s103
                                 + 0.0648495286073442 w_k1_AXP_s104
                                 + 0.0200153637198634 w_k1_AXP_s105
                                 + 0.0592019831886038 w_k1_AXP_s106
                                 + 0.0269763455854913 w_k1_AXP_s107
                                 + 0.0206392826993495 w_k1_AXP_s110
                                 + 0.0219741495099965 w_k1_AXP_s111
                                 + 0.0297613783776834 w_k1_AXP_s112
                                 + 0.0550585361507088 w_k1_AXP_s113
                                 + 0.0145491162544904 w_k1_AXP_s114
                                 + 0.0293833170319478 w_k1_AXP_s115
                                 + 0.0377200917197564 w_k1_AXP_s116
                                 + 0.0202160823959752 w_k1_AXP_s117
                                 + 0.0229089832944497 w_k1_DIS_s000
                                 + 0.0258530417042729 w_k1_DIS_s001
                                 + 0.0769234374827668 w_k1_DIS_s002
                                 + 0.0220963887969265 w_k1_DIS_s003
                                 + 0.037997635397021 w_k1_DIS_s004
                                 + 0.0175507311823309 w_k1_DIS_s005
                                 + 0.0326783386522444 w_k1_DIS_s006
                                 + 0.020347825428189 w_k1_DIS_s007
                                 + 0.0501906178790311 w_k1_DIS_s010
                                 + 0.0328096038983283 w_k1_DIS_s011
                                 + 0.0160016132648559 w_k1_DIS_s012
                                 + 0.0264877064273436 w_k1_DIS_s013
                                 + 0.0379234409152785 w_k1_DIS_s014
                                 + 0.0500656236785271 w_k1_DIS_s015
                                 + 0.0173060807723992 w_k1_DIS_s016
                                 + 0.0350911829601528 w_k1_DIS_s017
                                 + 0.0381558772618141 w_k1_DIS_s100
                                 + 0.0388766901945452 w_k1_DIS_s101
                                 + 0.0172602144280799 w_k1_DIS_s102
                                 + 0.0212607734878916 w_k1_DIS_s103
                                 + 0.0588246394878736 w_k1_DIS_s104
                                 + 0.0209507357690315 w_k1_DIS_s105
                                 + 0.0640036009550194 w_k1_DIS_s106
                                 + 0.026704107661956 w_k1_DIS_s107
                                 + 0.0212656403045152 w_k1_DIS_s110
                                 + 0.0213542142033342 w_k1_DIS_s111
                                 + 0.0322835082536747 w_k1_DIS_s112
                                 + 0.0549351965106269 w_k1_DIS_s113
                                 + 0.0133427794141069 w_k1_DIS_s114
                                 + 0.0281711773533449 w_k1_DIS_s115
                                 + 0.0392881786281755 w_k1_DIS_s116
                                 + 0.0193550626645297 w_k1_DIS_s117
                                 + 0.0225017417301094 w_k1_FDX_s000
                                 + 0.0231469108583142 w_k1_FDX_s001
                                 + 0.0739111835224287 w_k1_FDX_s002
                                 + 0.0219457687044775 w_k1_FDX_s003
                                 + 0.0356061066872141 w_k1_FDX_s004
                                 + 0.0171250247493461 w_k1_FDX_s005
                                 + 0.0326627355693439 w_k1_FDX_s006
                                 + 0.0211650379672493 w_k1_FDX_s007
                                 + 0.0497282113892099 w_k1_FDX_s010
                                 + 0.0326353099037037 w_k1_FDX_s011
                                 + 0.0156446821287679 w_k1_FDX_s012
                                 + 0.0259005383881534 w_k1_FDX_s013
                                 + 0.0388962646851344 w_k1_FDX_s014
                                 + 0.0507695716779691 w_k1_FDX_s015
                                 + 0.0163383048489431 w_k1_FDX_s016
                                 + 0.0358297396443312 w_k1_FDX_s017
                                 + 0.0381576364036171 w_k1_FDX_s100
                                 + 0.0370302925606482 w_k1_FDX_s101
                                 + 0.0170854637973513 w_k1_FDX_s102
                                 + 0.0216358920879131 w_k1_FDX_s103
                                 + 0.0603592487819162 w_k1_FDX_s104
                                 + 0.0199085467812823 w_k1_FDX_s105
                                 + 0.0584090678731453 w_k1_FDX_s106
                                 + 0.0260944208439461 w_k1_FDX_s107
                                 + 0.0196948160250069 w_k1_FDX_s110
                                 + 0.0215160845932174 w_k1_FDX_s111
                                 + 0.0313470608384595 w_k1_FDX_s112
                                 + 0.0508670117176484 w_k1_FDX_s113
                                 + 0.0129393175969442 w_k1_FDX_s114
                                 + 0.029043018416193 w_k1_FDX_s115
                                 + 0.0387574205554463 w_k1_FDX_s116
                                 + 0.0187158580778156 w_k1_FDX_s117
                                 + 0.0225600173057305 w_k1_IBM_s000
                                 + 0.0241177290015572 w_k1_IBM_s001
                                 + 0.0751440398859511 w_k1_IBM_s002
                                 + 0.0227098037497755 w_k1_IBM_s003
                                 + 0.0378947735008063 w_k1_IBM_s004
                                 + 0.0185691402107734 w_k1_IBM_s005
                                 + 0.0320781873813646 w_k1_IBM_s006
                                 + 0.0214405193461644 w_k1_IBM_s007
                                 + 0.0541212301157297 w_k1_IBM_s010
                                 + 0.032994781037508 w_k1_IBM_s011
                                 + 0.0154802837367315 w_k1_IBM_s012
                                 + 0.0258698249537226 w_k1_IBM_s013
                                 + 0.0385078764104431 w_k1_IBM_s014
                                 + 0.0506737887129212 w_k1_IBM_s015
                                 + 0.0173088066916766 w_k1_IBM_s016
                                 + 0.0369184263590762 w_k1_IBM_s017
                                 + 0.03645234957146 w_k1_IBM_s100
                                 + 0.036643927029674 w_k1_IBM_s101
                                 + 0.0171869801636228 w_k1_IBM_s102
                                 + 0.0208220312574265 w_k1_IBM_s103
                                 + 0.0575767100336944 w_k1_IBM_s104
                                 + 0.0202684703699759 w_k1_IBM_s105
                                 + 0.0619758014264783 w_k1_IBM_s106
                                 + 0.0260259749368523 w_k1_IBM_s107
                                 + 0.0201136610417907 w_k1_IBM_s110
                                 + 0.0221708556523339 w_k1_IBM_s111
                                 + 0.0326591947669536 w_k1_IBM_s112
                                 + 0.0499776462871558 w_k1_IBM_s113
                                 + 0.0138893108063477 w_k1_IBM_s114
                                 + 0.0293309011195154 w_k1_IBM_s115
                                 + 0.0381524903163447 w_k1_IBM_s116
                                 + 0.0199465852494076 w_k1_IBM_s117
                                 + 0.0270893701640081 w_k1_INTC_s000
                                 + 0.0256098184701961 w_k1_INTC_s001
                                 + 0.0649202025953524 w_k1_INTC_s002
                                 + 0.0218253242722007 w_k1_INTC_s003
                                 + 0.0351460622769552 w_k1_INTC_s004
                                 + 0.0166851413840955 w_k1_INTC_s005
                                 + 0.0329426876412156 w_k1_INTC_s006
                                 + 0.0223555137518773 w_k1_INTC_s007
                                 + 0.0499724579742028 w_k1_INTC_s010
                                 + 0.0358311409062496 w_k1_INTC_s011
                                 + 0.0152878350759637 w_k1_INTC_s012
                                 + 0.0204237550288997 w_k1_INTC_s013
                                 + 0.0419862282302696 w_k1_INTC_s014
                                 + 0.0509988208466423 w_k1_INTC_s015
                                 + 0.0174106397546821 w_k1_INTC_s016
                                 + 0.0389291072437159 w_k1_INTC_s017
                                 + 0.0362537250090015 w_k1_INTC_s100
                                 + 0.0338714492580295 w_k1_INTC_s101
                                 + 0.02106764196544 w_k1_INTC_s102
                                 + 0.0204990819115571 w_k1_INTC_s103
                                 + 0.0657536247580609 w_k1_INTC_s104
                                 + 0.0200846643060655 w_k1_INTC_s105
                                 + 0.062398383138317 w_k1_INTC_s106
                                 + 0.0254194547131349 w_k1_INTC_s107
                                 + 0.0220312465330437 w_k1_INTC_s110
                                 + 0.0185237422849744 w_k1_INTC_s111
                                 + 0.0295701697715441 w_k1_INTC_s112
                                 + 0.0542556584191792 w_k1_INTC_s113
                                 + 0.0132502144914925 w_k1_INTC_s114
                                 + 0.0282131078703802 w_k1_INTC_s115
                                 + 0.0369951775303943 w_k1_INTC_s116
                                 + 0.0180303773830621 w_k1_INTC_s117
                                 + 0.021439115734407 w_k1_KO_s000
                                 + 0.0251988240095319 w_k1_KO_s001
                                 + 0.0751360837981022 w_k1_KO_s002
                                 + 0.0205231183538216 w_k1_KO_s003
                                 + 0.0368071412177015 w_k1_KO_s004
                                 + 0.0166788692980642 w_k1_KO_s005
                                 + 0.0341938871582501 w_k1_KO_s006
                                 + 0.0204856164330111 w_k1_KO_s007
                                 + 0.0504446551771112 w_k1_KO_s010
                                 + 0.0339963195013102 w_k1_KO_s011
                                 + 0.0162008165719293 w_k1_KO_s012
                                 + 0.0252605854806834 w_k1_KO_s013
                                 + 0.0398021256992286 w_k1_KO_s014
                                 + 0.0448216754101445 w_k1_KO_s015
                                 + 0.0173963702284647 w_k1_KO_s016
                                 + 0.0356116954855363 w_k1_KO_s017
                                 + 0.03607734903842 w_k1_KO_s100
                                 + 0.0371232284383235 w_k1_KO_s101
                                 + 0.016092349439719 w_k1_KO_s102
                                 + 0.0197976582399396 w_k1_KO_s103
                                 + 0.0563978460315707 w_k1_KO_s104
                                 + 0.0212141025374485 w_k1_KO_s105
                                 + 0.0636447386066745 w_k1_KO_s106
                                 + 0.0251313737300491 w_k1_KO_s107
                                 + 0.0185507753928413 w_k1_KO_s110
                                 + 0.0218662241531581 w_k1_KO_s111
                                 + 0.0314995291935305 w_k1_KO_s112
                                 + 0.0465339524003924 w_k1_KO_s113
                                 + 0.0136179116704956 w_k1_KO_s114
                                 + 0.0309754087574393 w_k1_KO_s115
                                 + 0.0346339156380543 w_k1_KO_s116
                                 + 0.0186605817057288 w_k1_KO_s117
                                 + 0.0259791740342946 w_k1_MAT_s000
                                 + 0.0235942536265436 w_k1_MAT_s001
                                 + 0.0736782582414632 w_k1_MAT_s002
                                 + 0.0192564927662071 w_k1_MAT_s003
                                 + 0.0475442529738185 w_k1_MAT_s004
                                 + 0.0202550635423846 w_k1_MAT_s005
                                 + 0.0311566181765351 w_k1_MAT_s006
                                 + 0.0226345936371231 w_k1_MAT_s007
                                 + 0.0568819691338814 w_k1_MAT_s010
                                 + 0.0372168054584085 w_k1_MAT_s011
                                 + 0.0105609729995251 w_k1_MAT_s012
                                 + 0.0346164258634067 w_k1_MAT_s013
                                 + 0.0364281473262082 w_k1_MAT_s014
                                 + 0.0473729509248767 w_k1_MAT_s015
                                 + 0.0160484732457734 w_k1_MAT_s016
                                 + 0.03451122897563 w_k1_MAT_s017
                                 + 0.0362188171611697 w_k1_MAT_s100
                                 + 0.0381933474271887 w_k1_MAT_s101
                                 + 0.0174652109030572 w_k1_MAT_s102
                                 + 0.0225691029241747 w_k1_MAT_s103
                                 + 0.0746229936181655 w_k1_MAT_s104
                                 + 0.0247906644522538 w_k1_MAT_s105
                                 + 0.0597922963113307 w_k1_MAT_s106
                                 + 0.031012811419871 w_k1_MAT_s107
                                 + 0.0168961896654308 w_k1_MAT_s110
                                 + 0.0267725227725379 w_k1_MAT_s111
                                 + 0.0345103157060531 w_k1_MAT_s112
                                 + 0.0501521848115666 w_k1_MAT_s113
                                 + 0.0140802648320941 w_k1_MAT_s114
                                 + 0.0294907165408072 w_k1_MAT_s115
                                 + 0.0340723662870729 w_k1_MAT_s116
                                 + 0.0238873414753816 w_k1_MAT_s117
                                 + 0.0256040967786 w_k1_MSFT_s000
                                 + 0.0258784936620056 w_k1_MSFT_s001
                                 + 0.073131928415489 w_k1_MSFT_s002
                                 + 0.0228397544899763 w_k1_MSFT_s003
                                 + 0.0369052979332594 w_k1_MSFT_s004
                                 + 0.0172892094653853 w_k1_MSFT_s005
                                 + 0.0336585784384326 w_k1_MSFT_s006
                                 + 0.0229654820067485 w_k1_MSFT_s007
                                 + 0.0511705405520284 w_k1_MSFT_s010
                                 + 0.0337956830324742 w_k1_MSFT_s011
                                 + 0.015779554293976 w_k1_MSFT_s012
                                 + 0.0244932979919395 w_k1_MSFT_s013
                                 + 0.0375811407234121 w_k1_MSFT_s014
                                 + 0.0495841350195586 w_k1_MSFT_s015
                                 + 0.0171330741982608 w_k1_MSFT_s016
                                 + 0.0348733708846286 w_k1_MSFT_s017
                                 + 0.0358276443353557 w_k1_MSFT_s100
                                 + 0.0388528889248765 w_k1_MSFT_s101
                                 + 0.0168161799453232 w_k1_MSFT_s102
                                 + 0.0197176354120174 w_k1_MSFT_s103
                                 + 0.0587800385916499 w_k1_MSFT_s104
                                 + 0.0202655669315632 w_k1_MSFT_s105
                                 + 0.0651308372536812 w_k1_MSFT_s106
                                 + 0.0252065889535068 w_k1_MSFT_s107
                                 + 0.0189510086540588 w_k1_MSFT_s110
                                 + 0.0199964845344158 w_k1_MSFT_s111
                                 + 0.0308906096498175 w_k1_MSFT_s112
                                 + 0.047868629252012 w_k1_MSFT_s113
                                 + 0.0128734651859288 w_k1_MSFT_s114
                                 + 0.0316994118786831 w_k1_MSFT_s115
                                 + 0.037791448036802 w_k1_MSFT_s116
                                 + 0.0183518314682106 w_k1_MSFT_s117
                                 + 0.0224612894107485 w_k1_PG_s000
                                 + 0.0215220703292347 w_k1_PG_s001
                                 + 0.0758195483496542 w_k1_PG_s002
                                 + 0.0217697727204152 w_k1_PG_s003
                                 + 0.0337488243888805 w_k1_PG_s004
                                 + 0.0171857215767227 w_k1_PG_s005
                                 + 0.0289742955278373 w_k1_PG_s006
                                 + 0.0190806653340952 w_k1_PG_s007
                                 + 0.0500951557817181 w_k1_PG_s010
                                 + 0.0301370217257291 w_k1_PG_s011
                                 + 0.0154099584727 w_k1_PG_s012
                                 + 0.0251707582297493 w_k1_PG_s013
                                 + 0.0356867128575053 w_k1_PG_s014
                                 + 0.0552197044226668 w_k1_PG_s015
                                 + 0.0164428875312197 w_k1_PG_s016
                                 + 0.0362963885234154 w_k1_PG_s017
                                 + 0.0379961732661812 w_k1_PG_s100
                                 + 0.0354810343620373 w_k1_PG_s101
                                 + 0.0160388541978838 w_k1_PG_s102
                                 + 0.023275119614291 w_k1_PG_s103
                                 + 0.0586757689254796 w_k1_PG_s104
                                 + 0.0177596113766428 w_k1_PG_s105
                                 + 0.0588614729034999 w_k1_PG_s106
                                 + 0.0248758200926343 w_k1_PG_s107
                                 + 0.0197010547483138 w_k1_PG_s110
                                 + 0.0225478257692316 w_k1_PG_s111
                                 + 0.0330106738823489 w_k1_PG_s112
                                 + 0.0503386859411646 w_k1_PG_s113
                                 + 0.0139602542761671 w_k1_PG_s114
                                 + 0.0259034872689896 w_k1_PG_s115
                                 + 0.0378422171335321 w_k1_PG_s116
                                 + 0.0193293020405799 w_k1_PG_s117 >= 1.11
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
