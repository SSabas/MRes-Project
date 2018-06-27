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
 w_k1_ADM_s0_return_constraint:  1.11462 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.930017 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.01374 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.11283 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.794895 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13162 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.979555 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.884652 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.988016 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.645307 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.823894 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.830265 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.01089 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.14984 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.13859 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.955643 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.954101 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.775252 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.45283 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.927893 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.821686 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.18064 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.774211 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.923042 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.19875 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.944902 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.01187 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.812617 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.20705 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.24123 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.20605 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.20968 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0596 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.90396 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.850214 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.40028 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.16367 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.744442 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21596 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.884991 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.838816 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.25198 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.816904 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.15035 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.28165 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.09409 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.13985 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.769388 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.804211 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.963788 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.24664 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.62828 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.874663 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.963074 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.18307 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.592194 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.62088 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.688717 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.717667 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.00225 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.818421 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.935724 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.25463 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.33764 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.847061 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.520971 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.926739 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.49314 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.11673 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.37912 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.24255 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.793631 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.969041 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.63597 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.33074 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.970977 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26861 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.84392 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.777363 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.27951 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.791921 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.251 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.51043 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.01576 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.40149 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.799149 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.786333 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.874838 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.998821 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.52933 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.880009 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.05258 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.04457 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.596322 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.78075 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.711822 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.672198 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.28306 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.680752 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.891914 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.57327 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.29878 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.06416 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.525134 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.850876 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.16406 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.45969 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.35825 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.0844 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.810865 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.697679 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.64601 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.46923 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.11768 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16342 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.897467 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.878578 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.13735 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.814473 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.13491 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.23762 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.911843 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.30644 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.728738 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.890019 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.948001 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.14156 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.26713 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.02971 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.00335 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.967887 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.761404 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.497 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.861345 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.76975 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.25636 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.775488 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.94126 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.3398 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.07662 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.998758 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.709837 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.04672 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.1662 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.25439 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.15898 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.14227 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.939843 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.835917 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.54938 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.14034 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.917492 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.10051 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.977798 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.813451 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.20885 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.957778 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.05591 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.29569 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.46636 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.838124 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.36458 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.720063 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.02523 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.1502 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.7689 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.68006 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.14187 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.75416 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.767415 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.25765 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.508677 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.01648 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.809538 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.16261 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.918009 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.03435 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.67456 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.755648 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.640339 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.550049 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.23634 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.00016 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.46325 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.07246 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.670017 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.967405 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.02954 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 2.11289 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.47976 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.03427 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.885552 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.10986 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.20447 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.901953 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.14534 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.02075 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.829821 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.05616 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.763511 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.843596 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.903186 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.874605 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.21986 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.926198 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.949147 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.12626 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.81145 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.05296 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.913552 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.820794 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.22173 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.892182 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.889832 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.30324 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10227 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.966416 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.713439 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.973982 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.07428 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.18381 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.12355 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.13299 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.879326 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.802576 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.17103 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.995928 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.917106 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.08378 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.935672 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.901476 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.32192 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.873336 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.04153 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.2336 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.14295 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.95964 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.02547 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.737707 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.95652 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.0101 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.40832 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.819524 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.98931 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.06873 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.676833 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.11732 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.630171 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.824331 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.980378 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.934591 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.953449 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.18591 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.24321 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.897324 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.597172 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.819017 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.10563 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.3139 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.44469 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.13912 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.790012 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.954501 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.19334 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.64822 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.34428 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.1534 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.901115 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.740392 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.43639 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.921641 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.19594 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.5425 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.47333 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.96184 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.325 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.648979 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.27219 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.42847 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.98116 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.549557 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.06771 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.76017 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.608116 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.23168 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.480199 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.02807 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.64689 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.0197 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.960242 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.15444 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.91319 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.794561 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.486069 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.607906 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.30989 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.03616 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.42764 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.08269 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.645172 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.949749 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.17018 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 2.10145 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.48972 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11917 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.911442 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.926285 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.17912 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.819174 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.09901 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.19645 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.890219 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.29253 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.837795 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.942625 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.907508 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.998716 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.23111 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.887117 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.00875 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.04312 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.73293 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.15057 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.835509 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.832811 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2692 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.885238 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.970557 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.45539 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.05854 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.03958 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.721308 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.06591 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.07215 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.29537 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.16163 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.10181 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.974467 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.83453 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.31278 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.18195 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.05835 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08337 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.939753 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.900718 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10848 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.893679 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.07496 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12811 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.05348 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.03688 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.878251 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.918127 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.971172 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.0731 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.32197 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.901058 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.964163 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.992132 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.805067 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.27191 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.761483 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.899997 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.17362 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.911343 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.964249 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.19862 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.09624 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.969774 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.759546 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.878862 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.09067 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.06915 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.16791 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.12022 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.936317 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.94199 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.22514 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.3455 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.03086 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.180999117737005 z_k1_s0000
                                 - 0.15899987499855 z_k1_s0001
                                 - 0.16299994992364 z_k1_s0002
                                 - 0.126999995197546 z_k1_s0003
                                 - 0.123999983978711 z_k1_s0004
                                 - 0.11600001402846 z_k1_s0005
                                 - 0.495000052108519 z_k1_s0006
                                 - 0.433999584125631 z_k1_s0007
                                 - 0.381000529666202 z_k1_s0010
                                 - 0.244999187339444 z_k1_s0011
                                 - 0.117000031031386 z_k1_s0012
                                 - 0.225000488291021 z_k1_s0013
                                 - 0.228999591300863 z_k1_s0014
                                 - 0.0989999282881201 z_k1_s0015
                                 - 0.103999921450147 z_k1_s0016
                                 - 0.191999928461792 z_k1_s0017
                                 - 0.460999946883981 z_k1_s0020
                                 - 0.296999654320699 z_k1_s0021
                                 - 0.526000133607184 z_k1_s0022
                                 - 0.489998346684736 z_k1_s0023
                                 - 0.221999744962945 z_k1_s0024
                                 - 0.182999964528792 z_k1_s0025
                                 - 0.383999164519963 z_k1_s0026
                                 - 0.753000071487173 z_k1_s0027
                                 - 0.219999995864306 z_k1_s0030
                                 - 0.210000752869944 z_k1_s0031
                                 - 0.390999076059478 z_k1_s0032
                                 - 0.113999880129769 z_k1_s0033
                                 - 0.367999892173092 z_k1_s0034
                                 - 0.13299992181802 z_k1_s0035
                                 - 0.308999733509595 z_k1_s0036
                                 - 0.105000006435231 z_k1_s0037
                                 - 1.06299615897997 z_k1_s0040
                                 - 0.953996705472009 z_k1_s0041
                                 - 0.392999655096064 z_k1_s0042
                                 - 0.843997792848824 z_k1_s0043
                                 - 0.916997248225872 z_k1_s0044
                                 - 0.755999960749127 z_k1_s0045
                                 - 0.55399953857051 z_k1_s0046
                                 - 1.00899940678073 z_k1_s0047
                                 - 0.384999304925679 z_k1_s0050
                                 - 0.676000307122471 z_k1_s0051
                                 - 0.352000218904048 z_k1_s0052
                                 - 0.193000046396773 z_k1_s0053
                                 - 0.274000566851711 z_k1_s0054
                                 - 0.152000139835869 z_k1_s0055
                                 - 0.20900019227432 z_k1_s0056
                                 - 0.250999391502092 z_k1_s0057
                                 - 0.167999856027416 z_k1_s0060
                                 - 0.212000298855592 z_k1_s0061
                                 - 0.23200026082316 z_k1_s0062
                                 - 0.134999918780928 z_k1_s0063
                                 - 0.349999334432447 z_k1_s0064
                                 - 0.204999610167576 z_k1_s0065
                                 - 0.178000714510409 z_k1_s0066
                                 - 0.274999477054065 z_k1_s0067
                                 - 0.323999751509597 z_k1_s0070
                                 - 0.0969999959272297 z_k1_s0071
                                 - 0.206999683020079 z_k1_s0072
                                 - 0.304999326575556 z_k1_s0073
                                 - 0.110999904320884 z_k1_s0074
                                 - 0.164000595010762 z_k1_s0075
                                 - 0.0919999065866842 z_k1_s0076
                                 - 0.124000022686345 z_k1_s0077
                                 - 0.410002025711091 z_k1_s0100
                                 - 0.596000582364774 z_k1_s0101
                                 - 0.218000601094058 z_k1_s0102
                                 - 0.755999962541432 z_k1_s0103
                                 - 0.312000598348685 z_k1_s0104
                                 - 0.267000772674 z_k1_s0105
                                 - 0.354000797045899 z_k1_s0106
                                 - 0.701001982609909 z_k1_s0107
                                 - 0.576998509052451 z_k1_s0110
                                 - 0.793999536371009 z_k1_s0111
                                 - 0.895997761197436 z_k1_s0112
                                 - 0.927000007953623 z_k1_s0113
                                 - 0.836996962526461 z_k1_s0114
                                 - 0.231999411560065 z_k1_s0115
                                 - 0.269999336298316 z_k1_s0116
                                 - 0.366999569708382 z_k1_s0117
                                 - 0.558998089976841 z_k1_s0120
                                 - 0.452998759089843 z_k1_s0121
                                 - 0.422999881214606 z_k1_s0122
                                 - 0.242999634774844 z_k1_s0123
                                 - 0.445999214785297 z_k1_s0124
                                 - 0.242999634774844 z_k1_s0125
                                 - 0.384000758377769 z_k1_s0126
                                 - 0.156999749671014 z_k1_s0127
                                 - 0.136000075044701 z_k1_s0130
                                 - 0.536000972564646 z_k1_s0131
                                 - 0.170000037405856 z_k1_s0132
                                 - 0.342000696315529 z_k1_s0133
                                 - 0.203999999767011 z_k1_s0134
                                 - 0.260998987819202 z_k1_s0135
                                 - 0.168000092702277 z_k1_s0136
                                 - 0.43899970643969 z_k1_s0137
                                 - 0.137000490779143 z_k1_s0140
                                 - 0.456000882521318 z_k1_s0141
                                 - 0.670003040235436 z_k1_s0142
                                 - 0.397999642092415 z_k1_s0143
                                 - 0.217000578942358 z_k1_s0144
                                 - 0.195000562004993 z_k1_s0145
                                 - 0.626003006360706 z_k1_s0146
                                 - 0.224000597617918 z_k1_s0147
                                 - 0.107000307537093 z_k1_s0150
                                 - 0.131999770980012 z_k1_s0151
                                 - 0.1600001360366 z_k1_s0152
                                 - 0.163999588187219 z_k1_s0153
                                 - 0.147000079046102 z_k1_s0154
                                 - 0.173999793564612 z_k1_s0155
                                 - 0.112000410225789 z_k1_s0156
                                 - 0.0540000590373596 z_k1_s0157
                                 - 0.626997417520757 z_k1_s0160
                                 - 0.395998751221901 z_k1_s0161
                                 - 0.13799927870287 z_k1_s0162
                                 - 0.395998751221901 z_k1_s0163
                                 - 0.155999331985184 z_k1_s0164
                                 - 0.233998997977777 z_k1_s0165
                                 - 0.237998956018535 z_k1_s0166
                                 - 0.235998976998156 z_k1_s0167
                                 - 0.165999884534703 z_k1_s0170
                                 - 0.173999848549634 z_k1_s0171
                                 - 0.716000205985716 z_k1_s0172
                                 - 0.435000523123939 z_k1_s0173
                                 - 0.476001285537805 z_k1_s0174
                                 - 0.767998709632966 z_k1_s0175
                                 - 0.335999931426843 z_k1_s0176
                                 - 0.536000113899928 z_k1_s0177
                                 - 0.206999873855167 z_k1_s1000
                                 - 0.278000660443435 z_k1_s1001
                                 - 0.210000298809725 z_k1_s1002
                                 - 0.234000426445152 z_k1_s1003
                                 - 0.187999363810324 z_k1_s1004
                                 - 0.229000808854568 z_k1_s1005
                                 - 0.0970001068597204 z_k1_s1006
                                 - 0.193000617401428 z_k1_s1007
                                 - 0.225999807648896 z_k1_s1010
                                 - 0.521999592596088 z_k1_s1011
                                 - 0.607997130650599 z_k1_s1012
                                 - 0.620998158262157 z_k1_s1013
                                 - 0.722001512017899 z_k1_s1014
                                 - 0.648000292532315 z_k1_s1015
                                 - 0.822000156731011 z_k1_s1016
                                 - 0.460999400722061 z_k1_s1017
                                 - 0.446001336713858 z_k1_s1020
                                 - 0.246000047285381 z_k1_s1021
                                 - 0.143000040065491 z_k1_s1022
                                 - 0.359999451710056 z_k1_s1023
                                 - 0.210000102114376 z_k1_s1024
                                 - 0.228999955968469 z_k1_s1025
                                 - 0.572999900880496 z_k1_s1026
                                 - 0.60599921769532 z_k1_s1027
                                 - 0.902001603343628 z_k1_s1030
                                 - 0.323000452544351 z_k1_s1031
                                 - 0.545002359302018 z_k1_s1032
                                 - 0.355000539826217 z_k1_s1033
                                 - 0.696002025330798 z_k1_s1034
                                 - 0.818001183803405 z_k1_s1035
                                 - 1.1210008835704 z_k1_s1036
                                 - 0.318000645200662 z_k1_s1037
                                 - 0.360000064484643 z_k1_s1040
                                 - 0.290000951113494 z_k1_s1041
                                 - 0.25799993831389 z_k1_s1042
                                 - 0.0920001603462138 z_k1_s1043
                                 - 0.175000157230156 z_k1_s1044
                                 - 0.372000714034754 z_k1_s1045
                                 - 0.291999261036782 z_k1_s1046
                                 - 0.319000183024013 z_k1_s1047
                                 - 0.288000275508365 z_k1_s1050
                                 - 0.387000370214366 z_k1_s1051
                                 - 0.270999259244762 z_k1_s1052
                                 - 0.142000110840906 z_k1_s1053
                                 - 0.335000570470039 z_k1_s1054
                                 - 0.400000882650986 z_k1_s1055
                                 - 0.193000234628918 z_k1_s1056
                                 - 0.234000223850547 z_k1_s1057
                                 - 1.06099885034684 z_k1_s1060
                                 - 0.489998098032572 z_k1_s1061
                                 - 0.609001290496104 z_k1_s1062
                                 - 0.31100002394261 z_k1_s1063
                                 - 0.276000050711757 z_k1_s1064
                                 - 0.353999849243431 z_k1_s1065
                                 - 0.344999681589856 z_k1_s1066
                                 - 0.37199980275074 z_k1_s1067
                                 - 0.146999713668956 z_k1_s1070
                                 - 0.0910000424237825 z_k1_s1071
                                 - 0.229000611036001 z_k1_s1072
                                 - 0.0940000811762392 z_k1_s1073
                                 - 0.268000386417464 z_k1_s1074
                                 - 0.200000560162463 z_k1_s1075
                                 - 0.0930001087254466 z_k1_s1076
                                 - 0.0920000148745751 z_k1_s1077
                                 - 0.286000246803241 z_k1_s1100
                                 - 0.374999896874889 z_k1_s1101
                                 - 0.249999191583275 z_k1_s1102
                                 - 0.289998884716603 z_k1_s1103
                                 - 0.419998996899895 z_k1_s1104
                                 - 0.147999974093289 z_k1_s1105
                                 - 0.160999985311618 z_k1_s1106
                                 - 0.289000334738239 z_k1_s1107
                                 - 0.232999754260886 z_k1_s1110
                                 - 0.154999926834425 z_k1_s1111
                                 - 0.436999914391154 z_k1_s1112
                                 - 0.28899995408476 z_k1_s1113
                                 - 0.213999778313429 z_k1_s1114
                                 - 0.175999826418514 z_k1_s1115
                                 - 0.637999955266657 z_k1_s1116
                                 - 0.195999764184288 z_k1_s1117
                                 - 0.495999059816398 z_k1_s1120
                                 - 0.290000573118277 z_k1_s1121
                                 - 0.285999793765046 z_k1_s1122
                                 - 0.429999290485973 z_k1_s1123
                                 - 0.120000056600695 z_k1_s1124
                                 - 0.165999975164312 z_k1_s1125
                                 - 0.30100093133995 z_k1_s1126
                                 - 0.291000172956681 z_k1_s1127
                                 - 0.896998002798436 z_k1_s1130
                                 - 1.4049974388714 z_k1_s1131
                                 - 0.944997481510331 z_k1_s1132
                                 - 1.12100195030611 z_k1_s1133
                                 - 0.606998423878605 z_k1_s1134
                                 - 1.48099552215105 z_k1_s1135
                                 - 1.51799724005748 z_k1_s1136
                                 - 2.09999104536392 z_k1_s1137
                                 - 1.18200019351074 z_k1_s1140
                                 - 0.708998052154417 z_k1_s1141
                                 - 1.87099720000738 z_k1_s1142
                                 - 1.19800103003695 z_k1_s1143
                                 - 1.14599831132675 z_k1_s1144
                                 - 0.456998840635567 z_k1_s1145
                                 - 0.551998468421808 z_k1_s1146
                                 - 1.09899790752644 z_k1_s1147
                                 - 0.179999921585107 z_k1_s1150
                                 - 0.36600035139603 z_k1_s1151
                                 - 0.280998956526281 z_k1_s1152
                                 - 0.256000321916296 z_k1_s1153
                                 - 0.140000016410585 z_k1_s1154
                                 - 0.357998744962256 z_k1_s1155
                                 - 0.18399979600264 z_k1_s1156
                                 - 0.556998650530241 z_k1_s1157
                                 - 0.12199998560345 z_k1_s1160
                                 - 0.325999867078165 z_k1_s1161
                                 - 0.156999897919277 z_k1_s1162
                                 - 0.265999510880109 z_k1_s1163
                                 - 0.5009994286573 z_k1_s1164
                                 - 0.326999282082055 z_k1_s1165
                                 - 0.276000308912382 z_k1_s1166
                                 - 0.241000839796113 z_k1_s1167
                                 - 0.394998682616266 z_k1_s1170
                                 - 0.143999800823152 z_k1_s1171
                                 - 0.514999077385067 z_k1_s1172
                                 - 0.182999929123012 z_k1_s1173
                                 - 0.269000511107057 z_k1_s1174
                                 - 0.268000133984312 z_k1_s1175
                                 - 0.340000370845593 z_k1_s1176
                                 - 0.128999975776855 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.27108 w_k1_ADM_s000
                                 + 1.17883 w_k1_AXP_s000
                                 + 1.16229 w_k1_DIS_s000 + 1.0459 w_k1_FDX_s000
                                 + 1.31643 w_k1_IBM_s000
                                 + 0.884884 w_k1_INTC_s000
                                 + 1.19475 w_k1_KO_s000 + 1.29382 w_k1_MAT_s000
                                 + 1.10377 w_k1_MSFT_s000
                                 + 1.21872 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.1404 w_k1_ADM_s000
                                 + 1.14773 w_k1_AXP_s000
                                 + 1.54675 w_k1_DIS_s000
                                 + 1.35807 w_k1_FDX_s000
                                 + 1.04206 w_k1_IBM_s000
                                 + 1.05619 w_k1_INTC_s000
                                 + 1.22151 w_k1_KO_s000 + 1.24565 w_k1_MAT_s000
                                 + 1.23928 w_k1_MSFT_s000
                                 + 1.22322 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.13848 w_k1_ADM_s000
                                 + 0.827263 w_k1_AXP_s000
                                 + 1.09864 w_k1_DIS_s000
                                 + 1.03043 w_k1_FDX_s000
                                 + 0.935181 w_k1_IBM_s000
                                 + 1.04695 w_k1_INTC_s000
                                 + 1.06335 w_k1_KO_s000 + 1.04228 w_k1_MAT_s000
                                 + 1.14962 w_k1_MSFT_s000
                                 + 1.19511 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.06338 w_k1_ADM_s000
                                 + 1.30839 w_k1_AXP_s000
                                 + 1.36487 w_k1_DIS_s000
                                 + 1.17385 w_k1_FDX_s000
                                 + 1.64129 w_k1_IBM_s000
                                 + 1.13909 w_k1_INTC_s000
                                 + 1.35773 w_k1_KO_s000 + 1.57048 w_k1_MAT_s000
                                 + 1.12357 w_k1_MSFT_s000
                                 + 1.16877 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.20335 w_k1_ADM_s000
                                 + 1.19809 w_k1_AXP_s000
                                 + 1.03277 w_k1_DIS_s000
                                 + 1.28179 w_k1_FDX_s000
                                 + 0.897346 w_k1_IBM_s000
                                 + 1.06998 w_k1_INTC_s000 + 1.1103 w_k1_KO_s000
                                 + 1.09591 w_k1_MAT_s000
                                 + 1.11123 w_k1_MSFT_s000
                                 + 1.10106 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.57246 w_k1_ADM_s000
                                 + 1.50344 w_k1_AXP_s000
                                 + 1.05231 w_k1_DIS_s000
                                 + 1.20049 w_k1_FDX_s000
                                 + 0.752599 w_k1_IBM_s000
                                 + 1.06237 w_k1_INTC_s000
                                 + 0.946598 w_k1_KO_s000
                                 + 0.706144 w_k1_MAT_s000
                                 + 1.00465 w_k1_MSFT_s000
                                 + 1.08879 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.843849 w_k1_ADM_s000
                                 + 0.822386 w_k1_AXP_s000
                                 + 0.729006 w_k1_DIS_s000
                                 + 0.891372 w_k1_FDX_s000
                                 + 0.70516 w_k1_IBM_s000
                                 + 0.77834 w_k1_INTC_s000
                                 + 0.809307 w_k1_KO_s000
                                 + 0.732469 w_k1_MAT_s000
                                 + 0.881804 w_k1_MSFT_s000
                                 + 0.888146 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.16862 w_k1_ADM_s000
                                 + 0.981731 w_k1_AXP_s000
                                 + 0.845896 w_k1_DIS_s000
                                 + 0.907968 w_k1_FDX_s000
                                 + 1.11179 w_k1_IBM_s000
                                 + 1.0723 w_k1_INTC_s000 + 1.02977 w_k1_KO_s000
                                 + 0.996398 w_k1_MAT_s000
                                 + 0.876275 w_k1_MSFT_s000
                                 + 1.04677 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.908507 w_k1_ADM_s001
                                 + 0.844758 w_k1_AXP_s001
                                 + 0.995684 w_k1_DIS_s001
                                 + 0.99014 w_k1_FDX_s001
                                 + 0.707999 w_k1_IBM_s001
                                 + 1.02375 w_k1_INTC_s001
                                 + 0.778674 w_k1_KO_s001
                                 + 0.691941 w_k1_MAT_s001
                                 + 1.00702 w_k1_MSFT_s001
                                 + 0.826909 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.11168 w_k1_ADM_s001
                                 + 1.09888 w_k1_AXP_s001
                                 + 0.958676 w_k1_DIS_s001
                                 + 1.02646 w_k1_FDX_s001
                                 + 0.983502 w_k1_IBM_s001
                                 + 1.22525 w_k1_INTC_s001
                                 + 0.863004 w_k1_KO_s001
                                 + 0.984419 w_k1_MAT_s001
                                 + 0.908834 w_k1_MSFT_s001
                                 + 0.905743 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.29132 w_k1_ADM_s001
                                 + 1.36705 w_k1_AXP_s001 + 1.6176 w_k1_DIS_s001
                                 + 1.33535 w_k1_FDX_s001
                                 + 0.92006 w_k1_IBM_s001
                                 + 1.20855 w_k1_INTC_s001
                                 + 1.15329 w_k1_KO_s001
                                 + 0.921632 w_k1_MAT_s001
                                 + 1.25787 w_k1_MSFT_s001
                                 + 1.24792 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.03148 w_k1_ADM_s001
                                 + 0.933751 w_k1_AXP_s001
                                 + 1.36952 w_k1_DIS_s001
                                 + 1.12852 w_k1_FDX_s001
                                 + 1.20498 w_k1_IBM_s001
                                 + 1.21141 w_k1_INTC_s001
                                 + 1.14121 w_k1_KO_s001 + 1.01077 w_k1_MAT_s001
                                 + 1.15038 w_k1_MSFT_s001
                                 + 1.07675 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.92818 w_k1_ADM_s001
                                 + 0.673327 w_k1_AXP_s001
                                 + 0.834755 w_k1_DIS_s001
                                 + 0.898442 w_k1_FDX_s001
                                 + 0.972249 w_k1_IBM_s001
                                 + 1.02448 w_k1_INTC_s001
                                 + 0.86358 w_k1_KO_s001 + 0.82947 w_k1_MAT_s001
                                 + 0.948061 w_k1_MSFT_s001
                                 + 0.858595 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.32325 w_k1_ADM_s001
                                 + 1.21275 w_k1_AXP_s001
                                 + 1.40926 w_k1_DIS_s001
                                 + 1.20451 w_k1_FDX_s001
                                 + 1.73956 w_k1_IBM_s001
                                 + 1.12067 w_k1_INTC_s001
                                 + 1.52139 w_k1_KO_s001 + 1.74904 w_k1_MAT_s001
                                 + 1.33364 w_k1_MSFT_s001
                                 + 1.27817 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.08535 w_k1_ADM_s001
                                 + 1.37789 w_k1_AXP_s001
                                 + 1.07242 w_k1_DIS_s001
                                 + 1.23438 w_k1_FDX_s001
                                 + 1.28341 w_k1_IBM_s001
                                 + 0.997857 w_k1_INTC_s001
                                 + 1.09706 w_k1_KO_s001 + 1.2961 w_k1_MAT_s001
                                 + 1.00775 w_k1_MSFT_s001
                                 + 1.06991 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.856078 w_k1_ADM_s001
                                 + 0.859513 w_k1_AXP_s001
                                 + 0.910356 w_k1_DIS_s001
                                 + 0.915769 w_k1_FDX_s001
                                 + 1.2287 w_k1_IBM_s001
                                 + 0.996862 w_k1_INTC_s001
                                 + 0.961167 w_k1_KO_s001
                                 + 1.14116 w_k1_MAT_s001
                                 + 0.999995 w_k1_MSFT_s001
                                 + 0.970214 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.11677 w_k1_ADM_s002
                                 + 1.27903 w_k1_AXP_s002
                                 + 1.51271 w_k1_DIS_s002
                                 + 1.13022 w_k1_FDX_s002
                                 + 1.35382 w_k1_IBM_s002
                                 + 0.990657 w_k1_INTC_s002
                                 + 1.33092 w_k1_KO_s002 + 1.36673 w_k1_MAT_s002
                                 + 1.06169 w_k1_MSFT_s002
                                 + 1.21902 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.977799 w_k1_ADM_s002
                                 + 0.858248 w_k1_AXP_s002
                                 + 0.701456 w_k1_DIS_s002
                                 + 0.900849 w_k1_FDX_s002
                                 + 0.656385 w_k1_IBM_s002
                                 + 0.868565 w_k1_INTC_s002
                                 + 0.699602 w_k1_KO_s002
                                 + 0.572155 w_k1_MAT_s002
                                 + 0.889017 w_k1_MSFT_s002
                                 + 0.898485 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.782864 w_k1_ADM_s002
                                 + 0.66419 w_k1_AXP_s002
                                 + 0.693208 w_k1_DIS_s002
                                 + 0.738517 w_k1_FDX_s002
                                 + 1.05349 w_k1_IBM_s002
                                 + 0.898845 w_k1_INTC_s002
                                 + 0.96215 w_k1_KO_s002
                                 + 0.675752 w_k1_MAT_s002
                                 + 0.766694 w_k1_MSFT_s002
                                 + 0.913174 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.33099 w_k1_ADM_s002
                                 + 1.12583 w_k1_AXP_s002
                                 + 1.22007 w_k1_DIS_s002
                                 + 1.23305 w_k1_FDX_s002
                                 + 0.882251 w_k1_IBM_s002
                                 + 1.19105 w_k1_INTC_s002 + 1.098 w_k1_KO_s002
                                 + 0.867585 w_k1_MAT_s002
                                 + 1.06295 w_k1_MSFT_s002
                                 + 1.12866 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.1221 w_k1_ADM_s002
                                 + 1.14238 w_k1_AXP_s002
                                 + 1.08336 w_k1_DIS_s002
                                 + 1.05612 w_k1_FDX_s002
                                 + 1.14907 w_k1_IBM_s002
                                 + 1.00425 w_k1_INTC_s002
                                 + 1.32652 w_k1_KO_s002 + 1.221 w_k1_MAT_s002
                                 + 1.00916 w_k1_MSFT_s002
                                 + 1.23738 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.40274 w_k1_ADM_s002
                                 + 1.77607 w_k1_AXP_s002
                                 + 1.23866 w_k1_DIS_s002
                                 + 1.41717 w_k1_FDX_s002
                                 + 0.979636 w_k1_IBM_s002
                                 + 1.24524 w_k1_INTC_s002
                                 + 1.27394 w_k1_KO_s002 + 1.24491 w_k1_MAT_s002
                                 + 1.23458 w_k1_MSFT_s002
                                 + 1.33501 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.02074 w_k1_ADM_s002
                                 + 0.888544 w_k1_AXP_s002
                                 + 1.04084 w_k1_DIS_s002
                                 + 1.00559 w_k1_FDX_s002
                                 + 0.854794 w_k1_IBM_s002
                                 + 1.05558 w_k1_INTC_s002
                                 + 1.05572 w_k1_KO_s002
                                 + 0.779105 w_k1_MAT_s002
                                 + 1.07874 w_k1_MSFT_s002
                                 + 0.989873 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.975865 w_k1_ADM_s002
                                 + 0.996045 w_k1_AXP_s002
                                 + 0.820735 w_k1_DIS_s002
                                 + 0.875343 w_k1_FDX_s002
                                 + 1.09483 w_k1_IBM_s002
                                 + 0.836839 w_k1_INTC_s002
                                 + 1.02356 w_k1_KO_s002 + 1.23589 w_k1_MAT_s002
                                 + 0.790407 w_k1_MSFT_s002
                                 + 0.963187 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.4934 w_k1_ADM_s003
                                 + 1.40602 w_k1_AXP_s003 + 1.4183 w_k1_DIS_s003
                                 + 1.35058 w_k1_FDX_s003
                                 + 1.60087 w_k1_IBM_s003
                                 + 1.20743 w_k1_INTC_s003
                                 + 1.18806 w_k1_KO_s003 + 1.4415 w_k1_MAT_s003
                                 + 1.19123 w_k1_MSFT_s003
                                 + 1.21234 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.902019 w_k1_ADM_s003
                                 + 0.665518 w_k1_AXP_s003
                                 + 0.690622 w_k1_DIS_s003
                                 + 0.800773 w_k1_FDX_s003
                                 + 0.837922 w_k1_IBM_s003
                                 + 0.824748 w_k1_INTC_s003
                                 + 0.770062 w_k1_KO_s003
                                 + 0.570338 w_k1_MAT_s003
                                 + 0.753764 w_k1_MSFT_s003
                                 + 0.862857 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.24174 w_k1_ADM_s003
                                 + 0.965609 w_k1_AXP_s003
                                 + 0.992394 w_k1_DIS_s003
                                 + 1.0836 w_k1_FDX_s003
                                 + 0.932542 w_k1_IBM_s003
                                 + 1.01753 w_k1_INTC_s003
                                 + 1.05765 w_k1_KO_s003
                                 + 0.823858 w_k1_MAT_s003
                                 + 1.00799 w_k1_MSFT_s003
                                 + 1.09922 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.24594 w_k1_ADM_s003
                                 + 1.27443 w_k1_AXP_s003
                                 + 1.16257 w_k1_DIS_s003
                                 + 1.27246 w_k1_FDX_s003
                                 + 0.818876 w_k1_IBM_s003
                                 + 0.839144 w_k1_INTC_s003
                                 + 0.937687 w_k1_KO_s003
                                 + 0.956481 w_k1_MAT_s003
                                 + 1.04292 w_k1_MSFT_s003
                                 + 1.06215 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.24764 w_k1_ADM_s003
                                 + 1.03133 w_k1_AXP_s003
                                 + 0.954404 w_k1_DIS_s003
                                 + 0.855128 w_k1_FDX_s003
                                 + 1.26423 w_k1_IBM_s003
                                 + 0.761816 w_k1_INTC_s003
                                 + 1.08535 w_k1_KO_s003
                                 + 0.976212 w_k1_MAT_s003
                                 + 0.812553 w_k1_MSFT_s003
                                 + 0.987817 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.27367 w_k1_ADM_s003
                                 + 1.03895 w_k1_AXP_s003
                                 + 1.33933 w_k1_DIS_s003
                                 + 1.27737 w_k1_FDX_s003
                                 + 0.963661 w_k1_IBM_s003
                                 + 1.18135 w_k1_INTC_s003
                                 + 1.13999 w_k1_KO_s003
                                 + 0.949937 w_k1_MAT_s003
                                 + 1.15969 w_k1_MSFT_s003
                                 + 1.24313 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.894782 w_k1_ADM_s003
                                 + 0.723093 w_k1_AXP_s003
                                 + 0.749795 w_k1_DIS_s003
                                 + 0.90899 w_k1_FDX_s003
                                 + 0.688569 w_k1_IBM_s003
                                 + 0.871055 w_k1_INTC_s003
                                 + 0.775609 w_k1_KO_s003
                                 + 0.696364 w_k1_MAT_s003
                                 + 0.866275 w_k1_MSFT_s003
                                 + 0.86828 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.953192 w_k1_ADM_s003
                                 + 1.1311 w_k1_AXP_s003
                                 + 0.660533 w_k1_DIS_s003
                                 + 1.03552 w_k1_FDX_s003
                                 + 0.584162 w_k1_IBM_s003
                                 + 1.01999 w_k1_INTC_s003
                                 + 0.64542 w_k1_KO_s003
                                 + 0.679303 w_k1_MAT_s003
                                 + 0.820515 w_k1_MSFT_s003
                                 + 0.90058 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.08364 w_k1_ADM_s004
                                 + 1.4159 w_k1_AXP_s004 + 1.34105 w_k1_DIS_s004
                                 + 1.28603 w_k1_FDX_s004
                                 + 1.31828 w_k1_IBM_s004
                                 + 1.1552 w_k1_INTC_s004 + 1.17656 w_k1_KO_s004
                                 + 1.71813 w_k1_MAT_s004
                                 + 1.03686 w_k1_MSFT_s004
                                 + 1.14229 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.921079 w_k1_ADM_s004
                                 + 1.04285 w_k1_AXP_s004
                                 + 1.01971 w_k1_DIS_s004
                                 + 0.995139 w_k1_FDX_s004
                                 + 1.01621 w_k1_IBM_s004
                                 + 0.983382 w_k1_INTC_s004
                                 + 1.07559 w_k1_KO_s004 + 1.26985 w_k1_MAT_s004
                                 + 0.937406 w_k1_MSFT_s004
                                 + 1.01627 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.10637 w_k1_ADM_s004
                                 + 1.13513 w_k1_AXP_s004
                                 + 0.971833 w_k1_DIS_s004
                                 + 1.25161 w_k1_FDX_s004
                                 + 0.75235 w_k1_IBM_s004
                                 + 0.894046 w_k1_INTC_s004
                                 + 0.980398 w_k1_KO_s004
                                 + 1.13518 w_k1_MAT_s004
                                 + 0.994269 w_k1_MSFT_s004
                                 + 1.07922 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.967491 w_k1_ADM_s004
                                 + 0.954796 w_k1_AXP_s004
                                 + 1.01111 w_k1_DIS_s004
                                 + 0.919572 w_k1_FDX_s004
                                 + 1.44126 w_k1_IBM_s004
                                 + 0.890883 w_k1_INTC_s004
                                 + 1.22395 w_k1_KO_s004 + 1.9609 w_k1_MAT_s004
                                 + 0.938919 w_k1_MSFT_s004
                                 + 0.995799 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.32334 w_k1_ADM_s004
                                 + 1.18835 w_k1_AXP_s004
                                 + 1.50459 w_k1_DIS_s004
                                 + 1.32497 w_k1_FDX_s004
                                 + 0.91995 w_k1_IBM_s004
                                 + 0.978219 w_k1_INTC_s004
                                 + 1.17791 w_k1_KO_s004
                                 + 0.880306 w_k1_MAT_s004
                                 + 1.10304 w_k1_MSFT_s004 + 1.038 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.775151 w_k1_ADM_s004
                                 + 0.821132 w_k1_AXP_s004
                                 + 0.85796 w_k1_DIS_s004
                                 + 0.952797 w_k1_FDX_s004
                                 + 0.681719 w_k1_IBM_s004
                                 + 0.913954 w_k1_INTC_s004
                                 + 0.917734 w_k1_KO_s004
                                 + 0.875079 w_k1_MAT_s004
                                 + 0.87335 w_k1_MSFT_s004
                                 + 0.877775 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.03811 w_k1_ADM_s004
                                 + 1.08902 w_k1_AXP_s004
                                 + 0.80086 w_k1_DIS_s004
                                 + 0.898299 w_k1_FDX_s004
                                 + 0.867922 w_k1_IBM_s004
                                 + 0.801147 w_k1_INTC_s004
                                 + 0.809304 w_k1_KO_s004
                                 + 0.895621 w_k1_MAT_s004
                                 + 0.823888 w_k1_MSFT_s004
                                 + 0.88751 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.782681 w_k1_ADM_s004
                                 + 0.667059 w_k1_AXP_s004
                                 + 0.559946 w_k1_DIS_s004
                                 + 0.683703 w_k1_FDX_s004
                                 + 0.772471 w_k1_IBM_s004
                                 + 0.660359 w_k1_INTC_s004
                                 + 0.682858 w_k1_KO_s004
                                 + 0.844298 w_k1_MAT_s004
                                 + 0.628808 w_k1_MSFT_s004
                                 + 0.741915 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.773839 w_k1_ADM_s005
                                 + 0.769318 w_k1_AXP_s005
                                 + 0.685795 w_k1_DIS_s005
                                 + 0.777217 w_k1_FDX_s005
                                 + 1.06291 w_k1_IBM_s005
                                 + 0.655829 w_k1_INTC_s005
                                 + 0.921914 w_k1_KO_s005
                                 + 1.04657 w_k1_MAT_s005
                                 + 0.848189 w_k1_MSFT_s005
                                 + 0.935746 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.41069 w_k1_ADM_s005
                                 + 1.46027 w_k1_AXP_s005
                                 + 1.31505 w_k1_DIS_s005
                                 + 1.18047 w_k1_FDX_s005
                                 + 1.25602 w_k1_IBM_s005
                                 + 1.11422 w_k1_INTC_s005
                                 + 1.14761 w_k1_KO_s005 + 1.17391 w_k1_MAT_s005
                                 + 1.18733 w_k1_MSFT_s005 + 1.1858 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.935964 w_k1_ADM_s005
                                 + 0.923824 w_k1_AXP_s005
                                 + 1.14878 w_k1_DIS_s005
                                 + 0.960059 w_k1_FDX_s005
                                 + 1.26893 w_k1_IBM_s005
                                 + 0.926497 w_k1_INTC_s005
                                 + 1.15371 w_k1_KO_s005
                                 + 0.776802 w_k1_MAT_s005
                                 + 1.06768 w_k1_MSFT_s005 + 1.0715 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.10627 w_k1_ADM_s005
                                 + 0.933845 w_k1_AXP_s005
                                 + 0.760086 w_k1_DIS_s005
                                 + 0.883097 w_k1_FDX_s005
                                 + 0.674537 w_k1_IBM_s005
                                 + 1.01538 w_k1_INTC_s005
                                 + 0.833516 w_k1_KO_s005
                                 + 0.511726 w_k1_MAT_s005
                                 + 0.838675 w_k1_MSFT_s005
                                 + 0.862205 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.91957 w_k1_ADM_s005
                                 + 0.731568 w_k1_AXP_s005
                                 + 0.733865 w_k1_DIS_s005
                                 + 0.810727 w_k1_FDX_s005
                                 + 0.792314 w_k1_IBM_s005
                                 + 0.752842 w_k1_INTC_s005
                                 + 0.7028 w_k1_KO_s005 + 0.563799 w_k1_MAT_s005
                                 + 0.773787 w_k1_MSFT_s005
                                 + 0.94441 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.936677 w_k1_ADM_s005
                                 + 1.05653 w_k1_AXP_s005
                                 + 0.822699 w_k1_DIS_s005
                                 + 0.940929 w_k1_FDX_s005
                                 + 0.92865 w_k1_IBM_s005
                                 + 0.811437 w_k1_INTC_s005
                                 + 0.893259 w_k1_KO_s005
                                 + 0.729497 w_k1_MAT_s005
                                 + 0.834272 w_k1_MSFT_s005
                                 + 1.0086 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.14106 w_k1_ADM_s005
                                 + 0.891082 w_k1_AXP_s005
                                 + 1.17951 w_k1_DIS_s005
                                 + 1.04707 w_k1_FDX_s005
                                 + 0.887067 w_k1_IBM_s005
                                 + 1.01855 w_k1_INTC_s005
                                 + 1.03922 w_k1_KO_s005
                                 + 0.944614 w_k1_MAT_s005
                                 + 1.09524 w_k1_MSFT_s005 + 1.0336 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.23185 w_k1_ADM_s005
                                 + 1.32733 w_k1_AXP_s005
                                 + 1.21073 w_k1_DIS_s005
                                 + 1.16389 w_k1_FDX_s005
                                 + 0.860279 w_k1_IBM_s005
                                 + 1.01556 w_k1_INTC_s005
                                 + 0.926069 w_k1_KO_s005
                                 + 0.848061 w_k1_MAT_s005
                                 + 1.04299 w_k1_MSFT_s005
                                 + 1.12493 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.799484 w_k1_ADM_s006
                                 + 0.867867 w_k1_AXP_s006
                                 + 0.953871 w_k1_DIS_s006
                                 + 0.832157 w_k1_FDX_s006
                                 + 0.918267 w_k1_IBM_s006
                                 + 1.15036 w_k1_INTC_s006 + 1.0395 w_k1_KO_s006
                                 + 0.866786 w_k1_MAT_s006
                                 + 0.894243 w_k1_MSFT_s006
                                 + 0.940597 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.691217 w_k1_ADM_s006
                                 + 0.752417 w_k1_AXP_s006
                                 + 0.87466 w_k1_DIS_s006
                                 + 0.764169 w_k1_FDX_s006
                                 + 1.16121 w_k1_IBM_s006
                                 + 1.09654 w_k1_INTC_s006
                                 + 1.13111 w_k1_KO_s006
                                 + 0.970667 w_k1_MAT_s006
                                 + 0.928199 w_k1_MSFT_s006
                                 + 0.979094 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.759532 w_k1_ADM_s006
                                 + 1.20632 w_k1_AXP_s006
                                 + 1.28817 w_k1_DIS_s006
                                 + 1.09265 w_k1_FDX_s006
                                 + 1.52479 w_k1_IBM_s006
                                 + 1.23986 w_k1_INTC_s006
                                 + 1.27216 w_k1_KO_s006 + 1.60535 w_k1_MAT_s006
                                 + 1.16124 w_k1_MSFT_s006
                                 + 1.19209 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.978105 w_k1_ADM_s006
                                 + 1.74435 w_k1_AXP_s006 + 1.7042 w_k1_DIS_s006
                                 + 1.38435 w_k1_FDX_s006
                                 + 1.16385 w_k1_IBM_s006
                                 + 1.40721 w_k1_INTC_s006
                                 + 1.39223 w_k1_KO_s006 + 1.24414 w_k1_MAT_s006
                                 + 1.19351 w_k1_MSFT_s006
                                 + 1.19887 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.16803 w_k1_ADM_s006
                                 + 0.967287 w_k1_AXP_s006
                                 + 1.18621 w_k1_DIS_s006
                                 + 1.00769 w_k1_FDX_s006
                                 + 0.962776 w_k1_IBM_s006
                                 + 0.914456 w_k1_INTC_s006
                                 + 1.1282 w_k1_KO_s006 + 0.901592 w_k1_MAT_s006
                                 + 1.16814 w_k1_MSFT_s006
                                 + 1.06628 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.18818 w_k1_ADM_s006
                                 + 1.23985 w_k1_AXP_s006
                                 + 1.29628 w_k1_DIS_s006
                                 + 1.16563 w_k1_FDX_s006
                                 + 1.05888 w_k1_IBM_s006
                                 + 1.11163 w_k1_INTC_s006
                                 + 1.24134 w_k1_KO_s006 + 1.06801 w_k1_MAT_s006
                                 + 1.22169 w_k1_MSFT_s006
                                 + 1.14313 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.644405 w_k1_ADM_s006
                                 + 0.754467 w_k1_AXP_s006
                                 + 0.816153 w_k1_DIS_s006
                                 + 0.809034 w_k1_FDX_s006
                                 + 0.847957 w_k1_IBM_s006
                                 + 0.789878 w_k1_INTC_s006
                                 + 0.863152 w_k1_KO_s006
                                 + 0.92465 w_k1_MAT_s006
                                 + 0.812699 w_k1_MSFT_s006
                                 + 0.845662 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.00677 w_k1_ADM_s006
                                 + 0.996171 w_k1_AXP_s006
                                 + 0.826479 w_k1_DIS_s006
                                 + 0.96691 w_k1_FDX_s006
                                 + 0.744246 w_k1_IBM_s006
                                 + 0.899322 w_k1_INTC_s006
                                 + 0.719379 w_k1_KO_s006
                                 + 0.717074 w_k1_MAT_s006
                                 + 0.837911 w_k1_MSFT_s006
                                 + 0.878647 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.833022 w_k1_ADM_s007
                                 + 0.844768 w_k1_AXP_s007
                                 + 0.916795 w_k1_DIS_s007
                                 + 0.747083 w_k1_FDX_s007
                                 + 1.43145 w_k1_IBM_s007
                                 + 0.530586 w_k1_INTC_s007
                                 + 0.948994 w_k1_KO_s007 + 1.3322 w_k1_MAT_s007
                                 + 0.716626 w_k1_MSFT_s007
                                 + 0.769259 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.535234 w_k1_ADM_s007
                                 + 0.573548 w_k1_AXP_s007
                                 + 0.561747 w_k1_DIS_s007
                                 + 0.652401 w_k1_FDX_s007
                                 + 0.744379 w_k1_IBM_s007
                                 + 0.610741 w_k1_INTC_s007
                                 + 0.714779 w_k1_KO_s007
                                 + 0.896287 w_k1_MAT_s007
                                 + 0.720473 w_k1_MSFT_s007
                                 + 0.754117 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.8891 w_k1_ADM_s007
                                 + 0.672531 w_k1_AXP_s007
                                 + 1.01912 w_k1_DIS_s007
                                 + 0.880803 w_k1_FDX_s007
                                 + 0.844643 w_k1_IBM_s007
                                 + 0.792907 w_k1_INTC_s007
                                 + 0.893941 w_k1_KO_s007
                                 + 0.818033 w_k1_MAT_s007
                                 + 0.989457 w_k1_MSFT_s007
                                 + 0.865902 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.10068 w_k1_ADM_s007
                                 + 1.0629 w_k1_AXP_s007 + 1.41624 w_k1_DIS_s007
                                 + 1.21631 w_k1_FDX_s007
                                 + 1.08336 w_k1_IBM_s007
                                 + 0.959563 w_k1_INTC_s007
                                 + 0.96619 w_k1_KO_s007 + 1.20569 w_k1_MAT_s007
                                 + 1.01206 w_k1_MSFT_s007
                                 + 0.976154 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.18098 w_k1_ADM_s007
                                 + 1.01663 w_k1_AXP_s007
                                 + 0.805237 w_k1_DIS_s007
                                 + 1.04902 w_k1_FDX_s007
                                 + 0.685341 w_k1_IBM_s007
                                 + 0.958677 w_k1_INTC_s007
                                 + 0.938667 w_k1_KO_s007
                                 + 0.797318 w_k1_MAT_s007
                                 + 1.09985 w_k1_MSFT_s007
                                 + 0.86443 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.2737 w_k1_ADM_s007
                                 + 1.42576 w_k1_AXP_s007
                                 + 1.50749 w_k1_DIS_s007
                                 + 1.46144 w_k1_FDX_s007
                                 + 0.806227 w_k1_IBM_s007
                                 + 1.15776 w_k1_INTC_s007 + 1.1557 w_k1_KO_s007
                                 + 1.0596 w_k1_MAT_s007
                                 + 1.23298 w_k1_MSFT_s007
                                 + 1.19594 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.832127 w_k1_ADM_s007
                                 + 0.737597 w_k1_AXP_s007
                                 + 0.661673 w_k1_DIS_s007
                                 + 0.753437 w_k1_FDX_s007
                                 + 1.01255 w_k1_IBM_s007
                                 + 0.757878 w_k1_INTC_s007
                                 + 0.845137 w_k1_KO_s007
                                 + 1.09932 w_k1_MAT_s007
                                 + 0.841251 w_k1_MSFT_s007
                                 + 0.960606 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.714913 w_k1_ADM_s007
                                 + 1.32971 w_k1_AXP_s007
                                 + 1.05249 w_k1_DIS_s007
                                 + 1.04931 w_k1_FDX_s007
                                 + 1.08737 w_k1_IBM_s007
                                 + 1.10737 w_k1_INTC_s007
                                 + 1.11478 w_k1_KO_s007 + 1.07577 w_k1_MAT_s007
                                 + 0.867867 w_k1_MSFT_s007
                                 + 1.00954 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.02989 w_k1_ADM_s010
                                 + 0.84654 w_k1_AXP_s010
                                 + 0.92358 w_k1_DIS_s010
                                 + 1.01987 w_k1_FDX_s010
                                 + 0.745234 w_k1_IBM_s010
                                 + 0.897268 w_k1_INTC_s010
                                 + 0.807946 w_k1_KO_s010
                                 + 1.00015 w_k1_MAT_s010
                                 + 1.09187 w_k1_MSFT_s010
                                 + 1.07938 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.81251 w_k1_ADM_s010
                                 + 0.854709 w_k1_AXP_s010
                                 + 0.904858 w_k1_DIS_s010
                                 + 1.0758 w_k1_FDX_s010
                                 + 0.839173 w_k1_IBM_s010
                                 + 0.898432 w_k1_INTC_s010
                                 + 0.892909 w_k1_KO_s010
                                 + 0.996608 w_k1_MAT_s010
                                 + 1.03148 w_k1_MSFT_s010
                                 + 0.957765 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.09967 w_k1_ADM_s010
                                 + 1.15022 w_k1_AXP_s010 + 1.6557 w_k1_DIS_s010
                                 + 1.36348 w_k1_FDX_s010
                                 + 1.07149 w_k1_IBM_s010
                                 + 1.39431 w_k1_INTC_s010
                                 + 1.16546 w_k1_KO_s010
                                 + 0.963044 w_k1_MAT_s010
                                 + 1.50201 w_k1_MSFT_s010
                                 + 1.20214 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.876142 w_k1_ADM_s010
                                 + 0.823242 w_k1_AXP_s010
                                 + 0.831571 w_k1_DIS_s010
                                 + 0.806842 w_k1_FDX_s010
                                 + 1.06513 w_k1_IBM_s010
                                 + 0.907381 w_k1_INTC_s010
                                 + 0.818285 w_k1_KO_s010
                                 + 1.10798 w_k1_MAT_s010
                                 + 0.929218 w_k1_MSFT_s010
                                 + 0.955566 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.1574 w_k1_ADM_s010
                                 + 1.72787 w_k1_AXP_s010
                                 + 1.55505 w_k1_DIS_s010
                                 + 1.28351 w_k1_FDX_s010
                                 + 1.39895 w_k1_IBM_s010
                                 + 1.35336 w_k1_INTC_s010
                                 + 1.23632 w_k1_KO_s010 + 1.63816 w_k1_MAT_s010
                                 + 1.35151 w_k1_MSFT_s010
                                 + 1.25445 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.860184 w_k1_ADM_s010
                                 + 1.4719 w_k1_AXP_s010 + 1.04327 w_k1_DIS_s010
                                 + 1.07025 w_k1_FDX_s010
                                 + 1.48607 w_k1_IBM_s010
                                 + 1.17844 w_k1_INTC_s010 + 1.03 w_k1_KO_s010
                                 + 1.90188 w_k1_MAT_s010
                                 + 1.02396 w_k1_MSFT_s010
                                 + 1.12444 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.834163 w_k1_ADM_s010
                                 + 0.972236 w_k1_AXP_s010
                                 + 1.36417 w_k1_DIS_s010
                                 + 1.01505 w_k1_FDX_s010
                                 + 1.34944 w_k1_IBM_s010
                                 + 1.22298 w_k1_INTC_s010
                                 + 1.43528 w_k1_KO_s010 + 1.62891 w_k1_MAT_s010
                                 + 1.29167 w_k1_MSFT_s010
                                 + 1.09144 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.913512 w_k1_ADM_s010
                                 + 1.08882 w_k1_AXP_s010
                                 + 1.00946 w_k1_DIS_s010 + 0.977 w_k1_FDX_s010
                                 + 0.812243 w_k1_IBM_s010
                                 + 0.977436 w_k1_INTC_s010
                                 + 0.791736 w_k1_KO_s010
                                 + 0.79224 w_k1_MAT_s010
                                 + 1.0441 w_k1_MSFT_s010
                                 + 0.924618 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.964254 w_k1_ADM_s011
                                 + 0.864151 w_k1_AXP_s011
                                 + 0.788019 w_k1_DIS_s011
                                 + 0.87439 w_k1_FDX_s011
                                 + 1.17253 w_k1_IBM_s011
                                 + 0.883663 w_k1_INTC_s011
                                 + 1.02373 w_k1_KO_s011 + 1.07099 w_k1_MAT_s011
                                 + 0.866472 w_k1_MSFT_s011
                                 + 0.989575 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.33556 w_k1_ADM_s011
                                 + 1.24717 w_k1_AXP_s011
                                 + 1.35502 w_k1_DIS_s011
                                 + 1.26242 w_k1_FDX_s011
                                 + 1.07737 w_k1_IBM_s011
                                 + 1.22892 w_k1_INTC_s011
                                 + 1.38174 w_k1_KO_s011 + 1.25726 w_k1_MAT_s011
                                 + 1.19062 w_k1_MSFT_s011
                                 + 1.29984 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.25937 w_k1_ADM_s011
                                 + 1.17469 w_k1_AXP_s011
                                 + 0.856813 w_k1_DIS_s011
                                 + 1.06175 w_k1_FDX_s011
                                 + 0.888582 w_k1_IBM_s011
                                 + 1.00475 w_k1_INTC_s011
                                 + 1.01001 w_k1_KO_s011 + 1.16194 w_k1_MAT_s011
                                 + 0.919436 w_k1_MSFT_s011
                                 + 1.09442 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.819592 w_k1_ADM_s011
                                 + 0.645343 w_k1_AXP_s011
                                 + 0.659279 w_k1_DIS_s011
                                 + 0.78096 w_k1_FDX_s011
                                 + 0.74478 w_k1_IBM_s011
                                 + 0.957614 w_k1_INTC_s011
                                 + 0.792989 w_k1_KO_s011
                                 + 0.730353 w_k1_MAT_s011
                                 + 0.899724 w_k1_MSFT_s011
                                 + 0.894101 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.16243 w_k1_ADM_s011
                                 + 0.850461 w_k1_AXP_s011
                                 + 0.932345 w_k1_DIS_s011
                                 + 0.937863 w_k1_FDX_s011
                                 + 0.93257 w_k1_IBM_s011
                                 + 1.23466 w_k1_INTC_s011
                                 + 0.889529 w_k1_KO_s011
                                 + 0.741254 w_k1_MAT_s011
                                 + 1.03445 w_k1_MSFT_s011 + 1.0453 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.926405 w_k1_ADM_s011
                                 + 1.1777 w_k1_AXP_s011 + 1.00595 w_k1_DIS_s011
                                 + 0.924612 w_k1_FDX_s011
                                 + 1.66715 w_k1_IBM_s011
                                 + 1.05737 w_k1_INTC_s011
                                 + 1.20866 w_k1_KO_s011 + 1.22414 w_k1_MAT_s011
                                 + 0.950717 w_k1_MSFT_s011
                                 + 1.25943 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06577 w_k1_ADM_s011
                                 + 1.22835 w_k1_AXP_s011
                                 + 1.00643 w_k1_DIS_s011
                                 + 1.03824 w_k1_FDX_s011
                                 + 1.39854 w_k1_IBM_s011
                                 + 1.0791 w_k1_INTC_s011 + 1.29373 w_k1_KO_s011
                                 + 1.37042 w_k1_MAT_s011
                                 + 1.03507 w_k1_MSFT_s011
                                 + 1.15529 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.55146 w_k1_ADM_s011
                                 + 1.25591 w_k1_AXP_s011
                                 + 1.50164 w_k1_DIS_s011
                                 + 1.17041 w_k1_FDX_s011
                                 + 1.52884 w_k1_IBM_s011
                                 + 1.14028 w_k1_INTC_s011
                                 + 1.41091 w_k1_KO_s011 + 1.47996 w_k1_MAT_s011
                                 + 1.23681 w_k1_MSFT_s011
                                 + 1.34205 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.17041 w_k1_ADM_s012
                                 + 1.49982 w_k1_AXP_s012
                                 + 1.56299 w_k1_DIS_s012
                                 + 1.31243 w_k1_FDX_s012
                                 + 1.09511 w_k1_IBM_s012
                                 + 1.17396 w_k1_INTC_s012
                                 + 1.27748 w_k1_KO_s012 + 1.16736 w_k1_MAT_s012
                                 + 1.03377 w_k1_MSFT_s012
                                 + 1.14518 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.25339 w_k1_ADM_s012
                                 + 1.31073 w_k1_AXP_s012
                                 + 1.24608 w_k1_DIS_s012
                                 + 1.07756 w_k1_FDX_s012
                                 + 1.56343 w_k1_IBM_s012
                                 + 1.14074 w_k1_INTC_s012
                                 + 1.47979 w_k1_KO_s012 + 1.37952 w_k1_MAT_s012
                                 + 1.00122 w_k1_MSFT_s012
                                 + 1.23334 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.914169 w_k1_ADM_s012
                                 + 0.905455 w_k1_AXP_s012
                                 + 1.11793 w_k1_DIS_s012
                                 + 1.09228 w_k1_FDX_s012
                                 + 0.742605 w_k1_IBM_s012
                                 + 1.00222 w_k1_INTC_s012
                                 + 1.14084 w_k1_KO_s012
                                 + 0.533468 w_k1_MAT_s012
                                 + 1.07285 w_k1_MSFT_s012
                                 + 1.11187 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.828838 w_k1_ADM_s012
                                 + 0.604669 w_k1_AXP_s012
                                 + 0.649823 w_k1_DIS_s012
                                 + 0.701276 w_k1_FDX_s012
                                 + 0.911418 w_k1_IBM_s012
                                 + 0.682356 w_k1_INTC_s012
                                 + 0.986155 w_k1_KO_s012
                                 + 0.749231 w_k1_MAT_s012
                                 + 0.741944 w_k1_MSFT_s012
                                 + 1.01839 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.964213 w_k1_ADM_s012
                                 + 0.819273 w_k1_AXP_s012
                                 + 1.00035 w_k1_DIS_s012
                                 + 0.892911 w_k1_FDX_s012
                                 + 1.05585 w_k1_IBM_s012
                                 + 1.05356 w_k1_INTC_s012
                                 + 1.21079 w_k1_KO_s012
                                 + 0.755234 w_k1_MAT_s012
                                 + 0.936552 w_k1_MSFT_s012
                                 + 1.13826 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.744077 w_k1_ADM_s012
                                 + 1.04109 w_k1_AXP_s012
                                 + 0.843136 w_k1_DIS_s012
                                 + 0.928192 w_k1_FDX_s012
                                 + 1.11291 w_k1_IBM_s012
                                 + 0.895372 w_k1_INTC_s012
                                 + 1.01259 w_k1_KO_s012 + 1.34636 w_k1_MAT_s012
                                 + 0.835131 w_k1_MSFT_s012
                                 + 1.07952 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.906833 w_k1_ADM_s012
                                 + 0.882385 w_k1_AXP_s012
                                 + 0.723436 w_k1_DIS_s012
                                 + 0.905295 w_k1_FDX_s012
                                 + 0.723299 w_k1_IBM_s012
                                 + 0.840043 w_k1_INTC_s012
                                 + 0.898862 w_k1_KO_s012
                                 + 0.596271 w_k1_MAT_s012
                                 + 0.799507 w_k1_MSFT_s012
                                 + 0.961411 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.830064 w_k1_ADM_s012
                                 + 0.519246 w_k1_AXP_s012
                                 + 0.54878 w_k1_DIS_s012
                                 + 0.656002 w_k1_FDX_s012
                                 + 0.624188 w_k1_IBM_s012
                                 + 0.663891 w_k1_INTC_s012
                                 + 0.735579 w_k1_KO_s012
                                 + 0.429307 w_k1_MAT_s012
                                 + 0.604716 w_k1_MSFT_s012
                                 + 0.845977 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.32574 w_k1_ADM_s013
                                 + 1.37949 w_k1_AXP_s013
                                 + 1.62634 w_k1_DIS_s013
                                 + 1.10942 w_k1_FDX_s013
                                 + 1.65392 w_k1_IBM_s013
                                 + 1.1336 w_k1_INTC_s013 + 1.52347 w_k1_KO_s013
                                 + 1.69933 w_k1_MAT_s013
                                 + 1.30887 w_k1_MSFT_s013
                                 + 1.24814 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.49361 w_k1_ADM_s013
                                 + 1.21448 w_k1_AXP_s013
                                 + 1.32746 w_k1_DIS_s013
                                 + 1.20872 w_k1_FDX_s013
                                 + 0.908879 w_k1_IBM_s013
                                 + 1.18639 w_k1_INTC_s013
                                 + 1.01787 w_k1_KO_s013
                                 + 0.944438 w_k1_MAT_s013
                                 + 1.18894 w_k1_MSFT_s013
                                 + 1.14373 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.04257 w_k1_ADM_s013
                                 + 0.857099 w_k1_AXP_s013
                                 + 0.949477 w_k1_DIS_s013
                                 + 0.889775 w_k1_FDX_s013
                                 + 1.09034 w_k1_IBM_s013
                                 + 0.820251 w_k1_INTC_s013
                                 + 1.06126 w_k1_KO_s013
                                 + 0.880552 w_k1_MAT_s013
                                 + 1.06283 w_k1_MSFT_s013
                                 + 1.08412 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.07173 w_k1_ADM_s013
                                 + 1.19768 w_k1_AXP_s013
                                 + 0.974044 w_k1_DIS_s013
                                 + 0.997993 w_k1_FDX_s013
                                 + 0.893213 w_k1_IBM_s013
                                 + 1.16977 w_k1_INTC_s013
                                 + 1.00768 w_k1_KO_s013
                                 + 0.853821 w_k1_MAT_s013
                                 + 0.984771 w_k1_MSFT_s013
                                 + 0.966025 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.34429 w_k1_ADM_s013
                                 + 1.56206 w_k1_AXP_s013 + 1.5788 w_k1_DIS_s013
                                 + 1.19941 w_k1_FDX_s013
                                 + 1.22484 w_k1_IBM_s013
                                 + 1.05577 w_k1_INTC_s013
                                 + 1.36418 w_k1_KO_s013 + 1.33756 w_k1_MAT_s013
                                 + 1.147 w_k1_MSFT_s013 + 1.03172 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.858548 w_k1_ADM_s013
                                 + 0.964163 w_k1_AXP_s013
                                 + 1.00552 w_k1_DIS_s013
                                 + 0.871165 w_k1_FDX_s013
                                 + 1.34381 w_k1_IBM_s013
                                 + 0.730807 w_k1_INTC_s013
                                 + 0.9686 w_k1_KO_s013 + 1.28295 w_k1_MAT_s013
                                 + 0.903056 w_k1_MSFT_s013
                                 + 1.02161 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.688919 w_k1_ADM_s013
                                 + 0.851089 w_k1_AXP_s013
                                 + 0.675932 w_k1_DIS_s013
                                 + 0.657608 w_k1_FDX_s013
                                 + 0.994788 w_k1_IBM_s013
                                 + 0.968198 w_k1_INTC_s013
                                 + 0.848907 w_k1_KO_s013
                                 + 0.96122 w_k1_MAT_s013
                                 + 0.82067 w_k1_MSFT_s013
                                 + 0.923067 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.948879 w_k1_ADM_s013
                                 + 0.698411 w_k1_AXP_s013
                                 + 0.787049 w_k1_DIS_s013
                                 + 0.822093 w_k1_FDX_s013
                                 + 0.689465 w_k1_IBM_s013
                                 + 0.846017 w_k1_INTC_s013
                                 + 0.914003 w_k1_KO_s013
                                 + 0.661559 w_k1_MAT_s013
                                 + 1.02959 w_k1_MSFT_s013
                                 + 0.944181 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.23432 w_k1_ADM_s014
                                 + 1.45118 w_k1_AXP_s014
                                 + 0.865439 w_k1_DIS_s014
                                 + 1.04956 w_k1_FDX_s014
                                 + 0.825459 w_k1_IBM_s014
                                 + 1.3115 w_k1_INTC_s014
                                 + 0.951156 w_k1_KO_s014 + 1.1169 w_k1_MAT_s014
                                 + 1.03416 w_k1_MSFT_s014
                                 + 0.931457 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.739303 w_k1_ADM_s014
                                 + 0.670719 w_k1_AXP_s014
                                 + 0.708266 w_k1_DIS_s014
                                 + 0.787156 w_k1_FDX_s014
                                 + 0.703146 w_k1_IBM_s014
                                 + 1.02245 w_k1_INTC_s014
                                 + 0.830978 w_k1_KO_s014
                                 + 0.654754 w_k1_MAT_s014
                                 + 1.00606 w_k1_MSFT_s014
                                 + 0.837119 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.987277 w_k1_ADM_s014
                                 + 0.933743 w_k1_AXP_s014
                                 + 0.67986 w_k1_DIS_s014
                                 + 0.898471 w_k1_FDX_s014
                                 + 0.810138 w_k1_IBM_s014
                                 + 1.05432 w_k1_INTC_s014
                                 + 0.852327 w_k1_KO_s014
                                 + 0.772403 w_k1_MAT_s014
                                 + 0.90488 w_k1_MSFT_s014
                                 + 0.876115 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.27005 w_k1_ADM_s014
                                 + 1.25052 w_k1_AXP_s014
                                 + 1.67228 w_k1_DIS_s014
                                 + 1.32605 w_k1_FDX_s014
                                 + 1.50467 w_k1_IBM_s014
                                 + 1.78271 w_k1_INTC_s014
                                 + 1.87793 w_k1_KO_s014 + 1.40032 w_k1_MAT_s014
                                 + 1.70642 w_k1_MSFT_s014
                                 + 1.24778 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.17587 w_k1_ADM_s014
                                 + 0.965795 w_k1_AXP_s014
                                 + 1.40645 w_k1_DIS_s014 + 1.2315 w_k1_FDX_s014
                                 + 0.94479 w_k1_IBM_s014
                                 + 1.4412 w_k1_INTC_s014 + 1.38529 w_k1_KO_s014
                                 + 0.901334 w_k1_MAT_s014
                                 + 1.49485 w_k1_MSFT_s014 + 1.1188 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.06669 w_k1_ADM_s014
                                 + 1.31187 w_k1_AXP_s014 + 1.0669 w_k1_DIS_s014
                                 + 1.03241 w_k1_FDX_s014
                                 + 1.29399 w_k1_IBM_s014
                                 + 1.29502 w_k1_INTC_s014
                                 + 1.35526 w_k1_KO_s014 + 1.74629 w_k1_MAT_s014
                                 + 1.31483 w_k1_MSFT_s014
                                 + 1.08121 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.853569 w_k1_ADM_s014
                                 + 0.864284 w_k1_AXP_s014
                                 + 1.01826 w_k1_DIS_s014
                                 + 0.937787 w_k1_FDX_s014
                                 + 1.02761 w_k1_IBM_s014
                                 + 1.34058 w_k1_INTC_s014
                                 + 1.26254 w_k1_KO_s014 + 1.11216 w_k1_MAT_s014
                                 + 1.3001 w_k1_MSFT_s014
                                 + 0.933632 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.15694 w_k1_ADM_s014
                                 + 0.862769 w_k1_AXP_s014
                                 + 1.18612 w_k1_DIS_s014
                                 + 1.04587 w_k1_FDX_s014
                                 + 1.24032 w_k1_IBM_s014
                                 + 1.1842 w_k1_INTC_s014 + 1.43495 w_k1_KO_s014
                                 + 1.3001 w_k1_MAT_s014
                                 + 1.37632 w_k1_MSFT_s014
                                 + 1.08521 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.781935 w_k1_ADM_s015
                                 + 0.710997 w_k1_AXP_s015
                                 + 0.955337 w_k1_DIS_s015
                                 + 0.914049 w_k1_FDX_s015
                                 + 1.0391 w_k1_IBM_s015
                                 + 0.940181 w_k1_INTC_s015
                                 + 0.978855 w_k1_KO_s015
                                 + 0.767529 w_k1_MAT_s015
                                 + 1.10453 w_k1_MSFT_s015
                                 + 0.974633 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.775682 w_k1_ADM_s015
                                 + 0.739217 w_k1_AXP_s015
                                 + 0.556224 w_k1_DIS_s015
                                 + 0.810482 w_k1_FDX_s015
                                 + 0.834059 w_k1_IBM_s015
                                 + 0.775363 w_k1_INTC_s015
                                 + 0.695706 w_k1_KO_s015
                                 + 0.721401 w_k1_MAT_s015
                                 + 0.755201 w_k1_MSFT_s015
                                 + 0.837487 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.924834 w_k1_ADM_s015
                                 + 1.00044 w_k1_AXP_s015
                                 + 0.897258 w_k1_DIS_s015
                                 + 1.09201 w_k1_FDX_s015
                                 + 0.920497 w_k1_IBM_s015
                                 + 0.832896 w_k1_INTC_s015
                                 + 0.855846 w_k1_KO_s015
                                 + 0.917559 w_k1_MAT_s015
                                 + 1.00996 w_k1_MSFT_s015
                                 + 1.02442 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.905258 w_k1_ADM_s015
                                 + 1.01464 w_k1_AXP_s015
                                 + 1.08888 w_k1_DIS_s015 + 1.2525 w_k1_FDX_s015
                                 + 0.955227 w_k1_IBM_s015
                                 + 0.981144 w_k1_INTC_s015
                                 + 1.06042 w_k1_KO_s015 + 1.42713 w_k1_MAT_s015
                                 + 1.11638 w_k1_MSFT_s015
                                 + 1.12447 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.14541 w_k1_ADM_s015
                                 + 1.32938 w_k1_AXP_s015
                                 + 1.66568 w_k1_DIS_s015
                                 + 1.21999 w_k1_FDX_s015
                                 + 1.45458 w_k1_IBM_s015
                                 + 1.07606 w_k1_INTC_s015
                                 + 1.45997 w_k1_KO_s015 + 1.15584 w_k1_MAT_s015
                                 + 1.2464 w_k1_MSFT_s015 + 1.30089 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.899845 w_k1_ADM_s015
                                 + 0.932634 w_k1_AXP_s015
                                 + 0.713985 w_k1_DIS_s015
                                 + 0.854206 w_k1_FDX_s015
                                 + 1.04106 w_k1_IBM_s015
                                 + 0.759418 w_k1_INTC_s015
                                 + 0.89716 w_k1_KO_s015
                                 + 0.999346 w_k1_MAT_s015
                                 + 0.847332 w_k1_MSFT_s015
                                 + 0.934916 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.044 w_k1_ADM_s015
                                 + 1.07956 w_k1_AXP_s015
                                 + 1.05427 w_k1_DIS_s015
                                 + 1.00835 w_k1_FDX_s015
                                 + 1.07812 w_k1_IBM_s015
                                 + 0.989538 w_k1_INTC_s015
                                 + 0.948114 w_k1_KO_s015
                                 + 1.07296 w_k1_MAT_s015
                                 + 1.06423 w_k1_MSFT_s015
                                 + 1.06718 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.30534 w_k1_ADM_s015
                                 + 1.58761 w_k1_AXP_s015
                                 + 1.01061 w_k1_DIS_s015
                                 + 1.12528 w_k1_FDX_s015
                                 + 1.59569 w_k1_IBM_s015
                                 + 1.05337 w_k1_INTC_s015
                                 + 1.10563 w_k1_KO_s015 + 1.66313 w_k1_MAT_s015
                                 + 1.03264 w_k1_MSFT_s015
                                 + 1.06355 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.16902 w_k1_ADM_s016
                                 + 1.17725 w_k1_AXP_s016
                                 + 1.16372 w_k1_DIS_s016
                                 + 1.04614 w_k1_FDX_s016
                                 + 1.21139 w_k1_IBM_s016
                                 + 0.944879 w_k1_INTC_s016
                                 + 1.07962 w_k1_KO_s016 + 1.10999 w_k1_MAT_s016
                                 + 0.993765 w_k1_MSFT_s016
                                 + 1.13048 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.08352 w_k1_ADM_s016
                                 + 1.33593 w_k1_AXP_s016
                                 + 1.08847 w_k1_DIS_s016
                                 + 1.18146 w_k1_FDX_s016
                                 + 0.90932 w_k1_IBM_s016
                                 + 1.22861 w_k1_INTC_s016
                                 + 1.20174 w_k1_KO_s016
                                 + 0.940738 w_k1_MAT_s016
                                 + 1.0748 w_k1_MSFT_s016 + 1.05131 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.783361 w_k1_ADM_s016
                                 + 0.848549 w_k1_AXP_s016
                                 + 1.01276 w_k1_DIS_s016
                                 + 0.838971 w_k1_FDX_s016
                                 + 1.71693 w_k1_IBM_s016
                                 + 0.930651 w_k1_INTC_s016
                                 + 1.30854 w_k1_KO_s016 + 1.58795 w_k1_MAT_s016
                                 + 0.870378 w_k1_MSFT_s016
                                 + 1.14665 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.15587 w_k1_ADM_s016
                                 + 0.976662 w_k1_AXP_s016
                                 + 1.0456 w_k1_DIS_s016 + 1.08158 w_k1_FDX_s016
                                 + 0.787611 w_k1_IBM_s016
                                 + 0.942428 w_k1_INTC_s016
                                 + 0.88267 w_k1_KO_s016
                                 + 0.630157 w_k1_MAT_s016
                                 + 1.04551 w_k1_MSFT_s016
                                 + 0.934663 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.913028 w_k1_ADM_s016
                                 + 1.04334 w_k1_AXP_s016
                                 + 0.814045 w_k1_DIS_s016
                                 + 0.916532 w_k1_FDX_s016
                                 + 0.914324 w_k1_IBM_s016
                                 + 0.979007 w_k1_INTC_s016
                                 + 0.823655 w_k1_KO_s016
                                 + 0.583414 w_k1_MAT_s016
                                 + 0.876016 w_k1_MSFT_s016
                                 + 0.908079 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.986795 w_k1_ADM_s016
                                 + 0.704735 w_k1_AXP_s016
                                 + 0.664533 w_k1_DIS_s016
                                 + 0.790833 w_k1_FDX_s016
                                 + 0.792809 w_k1_IBM_s016
                                 + 0.781432 w_k1_INTC_s016
                                 + 0.817059 w_k1_KO_s016
                                 + 0.624569 w_k1_MAT_s016
                                 + 0.795059 w_k1_MSFT_s016
                                 + 0.840497 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.71846 w_k1_ADM_s016
                                 + 0.68502 w_k1_AXP_s016
                                 + 0.771715 w_k1_DIS_s016
                                 + 0.657106 w_k1_FDX_s016
                                 + 1.10099 w_k1_IBM_s016
                                 + 0.938324 w_k1_INTC_s016
                                 + 0.945177 w_k1_KO_s016
                                 + 0.856095 w_k1_MAT_s016
                                 + 0.716631 w_k1_MSFT_s016
                                 + 0.888408 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.15532 w_k1_ADM_s016
                                 + 1.53578 w_k1_AXP_s016
                                 + 1.52995 w_k1_DIS_s016
                                 + 1.27634 w_k1_FDX_s016 + 1.4692 w_k1_IBM_s016
                                 + 1.41525 w_k1_INTC_s016
                                 + 1.41073 w_k1_KO_s016 + 1.0605 w_k1_MAT_s016
                                 + 1.1769 w_k1_MSFT_s016 + 1.28368 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.946045 w_k1_ADM_s017
                                 + 1.48612 w_k1_AXP_s017
                                 + 0.934919 w_k1_DIS_s017
                                 + 1.03482 w_k1_FDX_s017
                                 + 0.993447 w_k1_IBM_s017
                                 + 0.999948 w_k1_INTC_s017
                                 + 0.976345 w_k1_KO_s017
                                 + 1.79249 w_k1_MAT_s017
                                 + 0.881288 w_k1_MSFT_s017
                                 + 1.05806 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.22665 w_k1_ADM_s017
                                 + 1.09915 w_k1_AXP_s017
                                 + 0.972989 w_k1_DIS_s017
                                 + 1.25823 w_k1_FDX_s017
                                 + 0.635446 w_k1_IBM_s017
                                 + 1.11487 w_k1_INTC_s017
                                 + 0.775361 w_k1_KO_s017
                                 + 0.658443 w_k1_MAT_s017
                                 + 1.17625 w_k1_MSFT_s017
                                 + 0.934905 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.08375 w_k1_ADM_s017
                                 + 1.15943 w_k1_AXP_s017
                                 + 1.01408 w_k1_DIS_s017
                                 + 1.04062 w_k1_FDX_s017
                                 + 0.778479 w_k1_IBM_s017
                                 + 1.00342 w_k1_INTC_s017
                                 + 0.884211 w_k1_KO_s017
                                 + 0.89817 w_k1_MAT_s017
                                 + 0.967612 w_k1_MSFT_s017
                                 + 0.824454 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.08303 w_k1_ADM_s017
                                 + 0.88169 w_k1_AXP_s017
                                 + 1.05755 w_k1_DIS_s017
                                 + 0.983742 w_k1_FDX_s017
                                 + 1.13562 w_k1_IBM_s017
                                 + 0.762574 w_k1_INTC_s017
                                 + 0.976922 w_k1_KO_s017 + 1.2005 w_k1_MAT_s017
                                 + 0.918135 w_k1_MSFT_s017
                                 + 0.955642 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.682194 w_k1_ADM_s017
                                 + 0.73452 w_k1_AXP_s017
                                 + 0.648517 w_k1_DIS_s017
                                 + 0.731603 w_k1_FDX_s017
                                 + 0.747338 w_k1_IBM_s017
                                 + 0.679376 w_k1_INTC_s017
                                 + 0.678452 w_k1_KO_s017
                                 + 0.706176 w_k1_MAT_s017
                                 + 0.665158 w_k1_MSFT_s017
                                 + 0.748033 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.34428 w_k1_ADM_s017
                                 + 1.44421 w_k1_AXP_s017
                                 + 1.28753 w_k1_DIS_s017
                                 + 1.25492 w_k1_FDX_s017
                                 + 1.34778 w_k1_IBM_s017
                                 + 1.10349 w_k1_INTC_s017
                                 + 1.22342 w_k1_KO_s017 + 1.24858 w_k1_MAT_s017
                                 + 1.14841 w_k1_MSFT_s017
                                 + 1.02678 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.933425 w_k1_ADM_s017
                                 + 1.01302 w_k1_AXP_s017
                                 + 1.33043 w_k1_DIS_s017
                                 + 0.969163 w_k1_FDX_s017 + 1.118 w_k1_IBM_s017
                                 + 1.01794 w_k1_INTC_s017
                                 + 1.21672 w_k1_KO_s017 + 1.16598 w_k1_MAT_s017
                                 + 0.936145 w_k1_MSFT_s017
                                 + 0.967412 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.860947 w_k1_ADM_s017
                                 + 0.984292 w_k1_AXP_s017
                                 + 0.762535 w_k1_DIS_s017
                                 + 0.88587 w_k1_FDX_s017
                                 + 1.20353 w_k1_IBM_s017
                                 + 0.842463 w_k1_INTC_s017
                                 + 0.98588 w_k1_KO_s017 + 1.25347 w_k1_MAT_s017
                                 + 0.793114 w_k1_MSFT_s017
                                 + 0.830881 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.303 w_k1_ADM_s100
                                 + 1.31627 w_k1_AXP_s100
                                 + 1.78705 w_k1_DIS_s100
                                 + 1.28198 w_k1_FDX_s100
                                 + 1.17094 w_k1_IBM_s100
                                 + 1.23324 w_k1_INTC_s100
                                 + 1.41542 w_k1_KO_s100 + 1.34146 w_k1_MAT_s100
                                 + 1.26042 w_k1_MSFT_s100 + 1.2409 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.41458 w_k1_ADM_s100
                                 + 0.945598 w_k1_AXP_s100
                                 + 1.32603 w_k1_DIS_s100 + 1.1409 w_k1_FDX_s100
                                 + 0.827873 w_k1_IBM_s100
                                 + 1.03202 w_k1_INTC_s100 + 1.0078 w_k1_KO_s100
                                 + 0.80068 w_k1_MAT_s100
                                 + 1.15398 w_k1_MSFT_s100
                                 + 0.890167 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.890829 w_k1_ADM_s100
                                 + 0.655759 w_k1_AXP_s100
                                 + 0.687722 w_k1_DIS_s100
                                 + 0.791846 w_k1_FDX_s100
                                 + 0.829727 w_k1_IBM_s100
                                 + 0.856096 w_k1_INTC_s100
                                 + 0.698605 w_k1_KO_s100
                                 + 0.911653 w_k1_MAT_s100
                                 + 0.843946 w_k1_MSFT_s100
                                 + 0.810012 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.842316 w_k1_ADM_s100
                                 + 1.28411 w_k1_AXP_s100
                                 + 1.16122 w_k1_DIS_s100
                                 + 1.06867 w_k1_FDX_s100
                                 + 1.54491 w_k1_IBM_s100
                                 + 1.10599 w_k1_INTC_s100
                                 + 1.12912 w_k1_KO_s100 + 1.53406 w_k1_MAT_s100
                                 + 1.01549 w_k1_MSFT_s100
                                 + 1.14244 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.08484 w_k1_ADM_s100
                                 + 1.06572 w_k1_AXP_s100
                                 + 1.27891 w_k1_DIS_s100
                                 + 1.07356 w_k1_FDX_s100
                                 + 1.04696 w_k1_IBM_s100 + 0.929 w_k1_INTC_s100
                                 + 1.0908 w_k1_KO_s100 + 1.13713 w_k1_MAT_s100
                                 + 0.930178 w_k1_MSFT_s100
                                 + 1.01534 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.96811 w_k1_ADM_s100
                                 + 1.10523 w_k1_AXP_s100
                                 + 0.927676 w_k1_DIS_s100
                                 + 0.812699 w_k1_FDX_s100
                                 + 1.02364 w_k1_IBM_s100
                                 + 0.789912 w_k1_INTC_s100
                                 + 0.932609 w_k1_KO_s100
                                 + 1.13386 w_k1_MAT_s100
                                 + 0.814711 w_k1_MSFT_s100
                                 + 0.942823 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.22898 w_k1_ADM_s100
                                 + 0.920351 w_k1_AXP_s100
                                 + 0.883044 w_k1_DIS_s100
                                 + 0.881592 w_k1_FDX_s100
                                 + 1.18773 w_k1_IBM_s100
                                 + 0.775466 w_k1_INTC_s100
                                 + 1.01843 w_k1_KO_s100 + 1.11546 w_k1_MAT_s100
                                 + 0.815037 w_k1_MSFT_s100
                                 + 1.03323 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.04411 w_k1_ADM_s100
                                 + 0.785142 w_k1_AXP_s100
                                 + 0.93195 w_k1_DIS_s100
                                 + 0.870421 w_k1_FDX_s100
                                 + 0.655613 w_k1_IBM_s100
                                 + 0.940167 w_k1_INTC_s100
                                 + 0.749534 w_k1_KO_s100
                                 + 0.675032 w_k1_MAT_s100
                                 + 0.845456 w_k1_MSFT_s100
                                 + 0.766918 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.744482 w_k1_ADM_s101
                                 + 0.578798 w_k1_AXP_s101
                                 + 0.48532 w_k1_DIS_s101
                                 + 0.663992 w_k1_FDX_s101
                                 + 0.68461 w_k1_IBM_s101
                                 + 0.977527 w_k1_INTC_s101
                                 + 0.594911 w_k1_KO_s101
                                 + 0.552591 w_k1_MAT_s101
                                 + 0.684301 w_k1_MSFT_s101
                                 + 0.745911 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.24638 w_k1_ADM_s101
                                 + 1.28999 w_k1_AXP_s101
                                 + 1.10373 w_k1_DIS_s101
                                 + 1.19964 w_k1_FDX_s101
                                 + 0.763214 w_k1_IBM_s101
                                 + 1.17387 w_k1_INTC_s101
                                 + 0.892344 w_k1_KO_s101
                                 + 0.798166 w_k1_MAT_s101
                                 + 1.04409 w_k1_MSFT_s101
                                 + 1.05341 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.0729 w_k1_ADM_s101
                                 + 0.946697 w_k1_AXP_s101
                                 + 0.724836 w_k1_DIS_s101
                                 + 0.908485 w_k1_FDX_s101
                                 + 0.690647 w_k1_IBM_s101
                                 + 1.05601 w_k1_INTC_s101
                                 + 0.808997 w_k1_KO_s101
                                 + 0.615609 w_k1_MAT_s101
                                 + 0.856766 w_k1_MSFT_s101
                                 + 0.811323 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.946996 w_k1_ADM_s101
                                 + 1.16234 w_k1_AXP_s101
                                 + 1.06542 w_k1_DIS_s101 + 1.0201 w_k1_FDX_s101
                                 + 1.39005 w_k1_IBM_s101
                                 + 1.2344 w_k1_INTC_s101 + 1.05107 w_k1_KO_s101
                                 + 1.45436 w_k1_MAT_s101
                                 + 0.994125 w_k1_MSFT_s101
                                 + 1.07108 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.29201 w_k1_ADM_s101
                                 + 1.23368 w_k1_AXP_s101
                                 + 1.04347 w_k1_DIS_s101
                                 + 1.10488 w_k1_FDX_s101
                                 + 0.962209 w_k1_IBM_s101
                                 + 1.33263 w_k1_INTC_s101
                                 + 1.02061 w_k1_KO_s101 + 1.03751 w_k1_MAT_s101
                                 + 1.15791 w_k1_MSFT_s101
                                 + 1.04485 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.836953 w_k1_ADM_s101
                                 + 0.933663 w_k1_AXP_s101
                                 + 0.822606 w_k1_DIS_s101
                                 + 0.848677 w_k1_FDX_s101
                                 + 1.03001 w_k1_IBM_s101
                                 + 0.903133 w_k1_INTC_s101
                                 + 0.865502 w_k1_KO_s101
                                 + 0.98245 w_k1_MAT_s101
                                 + 0.864718 w_k1_MSFT_s101
                                 + 0.909268 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.24321 w_k1_ADM_s101
                                 + 1.40969 w_k1_AXP_s101
                                 + 1.56702 w_k1_DIS_s101
                                 + 1.36887 w_k1_FDX_s101
                                 + 1.24843 w_k1_IBM_s101
                                 + 1.17799 w_k1_INTC_s101
                                 + 1.27359 w_k1_KO_s101 + 1.59931 w_k1_MAT_s101
                                 + 1.24121 w_k1_MSFT_s101
                                 + 1.24828 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.01813 w_k1_ADM_s101
                                 + 0.867335 w_k1_AXP_s101
                                 + 0.867454 w_k1_DIS_s101
                                 + 0.92804 w_k1_FDX_s101
                                 + 0.866707 w_k1_IBM_s101
                                 + 1.23543 w_k1_INTC_s101
                                 + 0.961642 w_k1_KO_s101
                                 + 0.947509 w_k1_MAT_s101
                                 + 1.07559 w_k1_MSFT_s101
                                 + 1.00083 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.59962 w_k1_ADM_s102
                                 + 1.28885 w_k1_AXP_s102
                                 + 1.14014 w_k1_DIS_s102
                                 + 1.15493 w_k1_FDX_s102
                                 + 0.905541 w_k1_IBM_s102
                                 + 1.05049 w_k1_INTC_s102
                                 + 1.02496 w_k1_KO_s102
                                 + 0.908582 w_k1_MAT_s102
                                 + 1.07604 w_k1_MSFT_s102
                                 + 1.08677 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.896925 w_k1_ADM_s102
                                 + 0.882242 w_k1_AXP_s102
                                 + 0.593448 w_k1_DIS_s102
                                 + 0.823886 w_k1_FDX_s102
                                 + 0.577359 w_k1_IBM_s102
                                 + 0.808679 w_k1_INTC_s102
                                 + 0.676076 w_k1_KO_s102
                                 + 0.603469 w_k1_MAT_s102
                                 + 0.698571 w_k1_MSFT_s102
                                 + 0.733387 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.00769 w_k1_ADM_s102
                                 + 0.838914 w_k1_AXP_s102
                                 + 0.993379 w_k1_DIS_s102
                                 + 1.0352 w_k1_FDX_s102
                                 + 0.694624 w_k1_IBM_s102
                                 + 0.964382 w_k1_INTC_s102
                                 + 0.934013 w_k1_KO_s102
                                 + 0.740135 w_k1_MAT_s102
                                 + 0.990443 w_k1_MSFT_s102
                                 + 0.957586 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.01007 w_k1_ADM_s102
                                 + 0.932351 w_k1_AXP_s102
                                 + 0.710935 w_k1_DIS_s102
                                 + 0.884656 w_k1_FDX_s102
                                 + 0.835365 w_k1_IBM_s102
                                 + 0.844013 w_k1_INTC_s102
                                 + 0.815306 w_k1_KO_s102
                                 + 0.909917 w_k1_MAT_s102
                                 + 0.821864 w_k1_MSFT_s102
                                 + 0.810059 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.09855 w_k1_ADM_s102
                                 + 1.41188 w_k1_AXP_s102
                                 + 1.07524 w_k1_DIS_s102
                                 + 1.25682 w_k1_FDX_s102
                                 + 1.04631 w_k1_IBM_s102
                                 + 1.34611 w_k1_INTC_s102
                                 + 1.09557 w_k1_KO_s102 + 1.18566 w_k1_MAT_s102
                                 + 1.17253 w_k1_MSFT_s102
                                 + 1.12985 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.975952 w_k1_ADM_s102
                                 + 0.942551 w_k1_AXP_s102
                                 + 0.988382 w_k1_DIS_s102
                                 + 0.995159 w_k1_FDX_s102
                                 + 0.865986 w_k1_IBM_s102
                                 + 0.94272 w_k1_INTC_s102
                                 + 1.01589 w_k1_KO_s102 + 1.13719 w_k1_MAT_s102
                                 + 1.03487 w_k1_MSFT_s102
                                 + 1.01321 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.51023 w_k1_ADM_s102
                                 + 1.5544 w_k1_AXP_s102 + 1.47334 w_k1_DIS_s102
                                 + 1.15262 w_k1_FDX_s102
                                 + 1.59078 w_k1_IBM_s102
                                 + 1.26686 w_k1_INTC_s102
                                 + 1.61768 w_k1_KO_s102 + 1.66027 w_k1_MAT_s102
                                 + 1.10598 w_k1_MSFT_s102
                                 + 1.14428 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.14681 w_k1_ADM_s102
                                 + 0.983145 w_k1_AXP_s102
                                 + 0.844953 w_k1_DIS_s102
                                 + 0.868089 w_k1_FDX_s102
                                 + 1.19492 w_k1_IBM_s102
                                 + 0.824235 w_k1_INTC_s102
                                 + 0.999283 w_k1_KO_s102 + 1.1845 w_k1_MAT_s102
                                 + 0.759346 w_k1_MSFT_s102
                                 + 0.952068 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.51902 w_k1_ADM_s103
                                 + 1.16505 w_k1_AXP_s103
                                 + 1.53531 w_k1_DIS_s103
                                 + 1.29189 w_k1_FDX_s103
                                 + 1.52422 w_k1_IBM_s103
                                 + 1.00525 w_k1_INTC_s103
                                 + 1.51891 w_k1_KO_s103 + 1.67718 w_k1_MAT_s103
                                 + 1.35715 w_k1_MSFT_s103
                                 + 1.30923 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.29379 w_k1_ADM_s103
                                 + 0.684688 w_k1_AXP_s103
                                 + 0.687936 w_k1_DIS_s103
                                 + 0.915962 w_k1_FDX_s103
                                 + 0.56323 w_k1_IBM_s103
                                 + 0.823335 w_k1_INTC_s103
                                 + 0.794565 w_k1_KO_s103
                                 + 0.480191 w_k1_MAT_s103
                                 + 0.978665 w_k1_MSFT_s103
                                 + 0.927608 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.959604 w_k1_ADM_s103
                                 + 0.824567 w_k1_AXP_s103
                                 + 0.754451 w_k1_DIS_s103
                                 + 0.809502 w_k1_FDX_s103
                                 + 0.825542 w_k1_IBM_s103
                                 + 0.897931 w_k1_INTC_s103
                                 + 0.914833 w_k1_KO_s103
                                 + 0.866669 w_k1_MAT_s103
                                 + 0.943898 w_k1_MSFT_s103
                                 + 0.935615 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.795715 w_k1_ADM_s103
                                 + 0.526888 w_k1_AXP_s103
                                 + 0.587235 w_k1_DIS_s103
                                 + 0.703551 w_k1_FDX_s103
                                 + 0.658303 w_k1_IBM_s103
                                 + 0.675404 w_k1_INTC_s103
                                 + 0.758435 w_k1_KO_s103
                                 + 0.524159 w_k1_MAT_s103
                                 + 0.756357 w_k1_MSFT_s103
                                 + 0.80713 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.37033 w_k1_ADM_s103
                                 + 0.736706 w_k1_AXP_s103
                                 + 1.02822 w_k1_DIS_s103
                                 + 1.03286 w_k1_FDX_s103
                                 + 0.747019 w_k1_IBM_s103
                                 + 0.804199 w_k1_INTC_s103
                                 + 0.981846 w_k1_KO_s103
                                 + 0.69068 w_k1_MAT_s103
                                 + 1.10911 w_k1_MSFT_s103
                                 + 1.00526 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.69778 w_k1_ADM_s103
                                 + 1.01418 w_k1_AXP_s103
                                 + 1.23547 w_k1_DIS_s103 + 1.2437 w_k1_FDX_s103
                                 + 0.856138 w_k1_IBM_s103
                                 + 1.07575 w_k1_INTC_s103
                                 + 1.24385 w_k1_KO_s103
                                 + 0.770467 w_k1_MAT_s103
                                 + 1.25444 w_k1_MSFT_s103
                                 + 1.04883 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.07706 w_k1_ADM_s103
                                 + 0.983811 w_k1_AXP_s103
                                 + 0.948651 w_k1_DIS_s103
                                 + 0.947267 w_k1_FDX_s103
                                 + 1.2178 w_k1_IBM_s103
                                 + 0.964645 w_k1_INTC_s103
                                 + 1.0929 w_k1_KO_s103 + 1.10662 w_k1_MAT_s103
                                 + 1.08727 w_k1_MSFT_s103 + 1.0046 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.812353 w_k1_ADM_s103
                                 + 0.736815 w_k1_AXP_s103
                                 + 0.602208 w_k1_DIS_s103
                                 + 0.694556 w_k1_FDX_s103
                                 + 0.993266 w_k1_IBM_s103
                                 + 0.754761 w_k1_INTC_s103
                                 + 0.826671 w_k1_KO_s103
                                 + 0.831224 w_k1_MAT_s103
                                 + 0.745179 w_k1_MSFT_s103
                                 + 0.864646 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.91114 w_k1_ADM_s104
                                 + 0.850125 w_k1_AXP_s104
                                 + 0.927359 w_k1_DIS_s104
                                 + 0.936397 w_k1_FDX_s104
                                 + 0.777964 w_k1_IBM_s104
                                 + 1.04457 w_k1_INTC_s104
                                 + 0.918774 w_k1_KO_s104
                                 + 0.586949 w_k1_MAT_s104
                                 + 0.993089 w_k1_MSFT_s104
                                 + 0.902109 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.986437 w_k1_ADM_s104
                                 + 0.810268 w_k1_AXP_s104
                                 + 0.814747 w_k1_DIS_s104
                                 + 0.92335 w_k1_FDX_s104
                                 + 0.955933 w_k1_IBM_s104
                                 + 0.999426 w_k1_INTC_s104
                                 + 1.00474 w_k1_KO_s104 + 1.02017 w_k1_MAT_s104
                                 + 1.02561 w_k1_MSFT_s104
                                 + 0.945008 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.943665 w_k1_ADM_s104
                                 + 0.874032 w_k1_AXP_s104
                                 + 1.02646 w_k1_DIS_s104
                                 + 0.989617 w_k1_FDX_s104
                                 + 1.44498 w_k1_IBM_s104
                                 + 1.11249 w_k1_INTC_s104
                                 + 1.37539 w_k1_KO_s104 + 1.17663 w_k1_MAT_s104
                                 + 1.15451 w_k1_MSFT_s104
                                 + 1.04702 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.13509 w_k1_ADM_s104
                                 + 1.10847 w_k1_AXP_s104
                                 + 0.882526 w_k1_DIS_s104
                                 + 1.01014 w_k1_FDX_s104
                                 + 1.10935 w_k1_IBM_s104
                                 + 0.91956 w_k1_INTC_s104
                                 + 1.01956 w_k1_KO_s104 + 1.10077 w_k1_MAT_s104
                                 + 0.960042 w_k1_MSFT_s104
                                 + 1.08826 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.922509 w_k1_ADM_s104
                                 + 0.658004 w_k1_AXP_s104
                                 + 0.529428 w_k1_DIS_s104
                                 + 0.756379 w_k1_FDX_s104
                                 + 0.702577 w_k1_IBM_s104
                                 + 0.751682 w_k1_INTC_s104
                                 + 0.675523 w_k1_KO_s104
                                 + 0.855883 w_k1_MAT_s104
                                 + 0.737678 w_k1_MSFT_s104
                                 + 0.779661 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.31094 w_k1_ADM_s104
                                 + 1.26604 w_k1_AXP_s104
                                 + 1.35287 w_k1_DIS_s104
                                 + 1.37488 w_k1_FDX_s104
                                 + 1.00353 w_k1_IBM_s104
                                 + 1.32604 w_k1_INTC_s104
                                 + 1.28141 w_k1_KO_s104
                                 + 0.875456 w_k1_MAT_s104
                                 + 1.17652 w_k1_MSFT_s104
                                 + 1.15853 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.1204 w_k1_ADM_s104
                                 + 1.21691 w_k1_AXP_s104
                                 + 1.35247 w_k1_DIS_s104
                                 + 1.22381 w_k1_FDX_s104
                                 + 1.19247 w_k1_IBM_s104
                                 + 1.43706 w_k1_INTC_s104 + 1.1584 w_k1_KO_s104
                                 + 1.15249 w_k1_MAT_s104
                                 + 1.31413 w_k1_MSFT_s104
                                 + 1.17753 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.11439 w_k1_ADM_s104
                                 + 1.07001 w_k1_AXP_s104
                                 + 1.01498 w_k1_DIS_s104
                                 + 1.04182 w_k1_FDX_s104
                                 + 1.27324 w_k1_IBM_s104
                                 + 1.14849 w_k1_INTC_s104
                                 + 1.13733 w_k1_KO_s104 + 0.8918 w_k1_MAT_s104
                                 + 1.04232 w_k1_MSFT_s104 + 0.9882 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.906258 w_k1_ADM_s105
                                 + 0.861695 w_k1_AXP_s105
                                 + 0.862418 w_k1_DIS_s105
                                 + 0.870731 w_k1_FDX_s105
                                 + 0.890266 w_k1_IBM_s105
                                 + 1.05529 w_k1_INTC_s105
                                 + 0.987903 w_k1_KO_s105
                                 + 0.648007 w_k1_MAT_s105
                                 + 0.941312 w_k1_MSFT_s105
                                 + 0.927431 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.956945 w_k1_ADM_s105
                                 + 1.25334 w_k1_AXP_s105
                                 + 1.09825 w_k1_DIS_s105
                                 + 1.03052 w_k1_FDX_s105
                                 + 1.05476 w_k1_IBM_s105
                                 + 1.07828 w_k1_INTC_s105
                                 + 1.15945 w_k1_KO_s105 + 1.60687 w_k1_MAT_s105
                                 + 1.11361 w_k1_MSFT_s105
                                 + 1.13063 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.789419 w_k1_ADM_s105
                                 + 0.822802 w_k1_AXP_s105
                                 + 0.635802 w_k1_DIS_s105
                                 + 0.757557 w_k1_FDX_s105
                                 + 0.655529 w_k1_IBM_s105
                                 + 0.921306 w_k1_INTC_s105
                                 + 0.782538 w_k1_KO_s105
                                 + 0.737296 w_k1_MAT_s105
                                 + 0.848961 w_k1_MSFT_s105
                                 + 0.83232 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.15862 w_k1_ADM_s105
                                 + 1.83853 w_k1_AXP_s105
                                 + 1.73135 w_k1_DIS_s105
                                 + 1.36924 w_k1_FDX_s105
                                 + 1.03785 w_k1_IBM_s105
                                 + 1.31007 w_k1_INTC_s105
                                 + 1.14031 w_k1_KO_s105 + 1.24602 w_k1_MAT_s105
                                 + 1.48123 w_k1_MSFT_s105
                                 + 1.07404 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.855318 w_k1_ADM_s105
                                 + 1.12885 w_k1_AXP_s105
                                 + 0.768675 w_k1_DIS_s105
                                 + 0.828159 w_k1_FDX_s105
                                 + 1.31491 w_k1_IBM_s105
                                 + 1.05726 w_k1_INTC_s105
                                 + 1.03408 w_k1_KO_s105 + 1.78493 w_k1_MAT_s105
                                 + 0.93952 w_k1_MSFT_s105
                                 + 1.01832 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.13659 w_k1_ADM_s105
                                 + 1.34977 w_k1_AXP_s105 + 1.3458 w_k1_DIS_s105
                                 + 1.15637 w_k1_FDX_s105
                                 + 1.52426 w_k1_IBM_s105
                                 + 1.12452 w_k1_INTC_s105
                                 + 1.42257 w_k1_KO_s105 + 1.42636 w_k1_MAT_s105
                                 + 1.3561 w_k1_MSFT_s105 + 1.1819 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.00806 w_k1_ADM_s105
                                 + 1.06511 w_k1_AXP_s105
                                 + 1.29945 w_k1_DIS_s105
                                 + 1.15011 w_k1_FDX_s105
                                 + 0.803572 w_k1_IBM_s105
                                 + 1.12702 w_k1_INTC_s105
                                 + 1.07514 w_k1_KO_s105 + 1.15614 w_k1_MAT_s105
                                 + 1.22703 w_k1_MSFT_s105
                                 + 1.10686 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.920483 w_k1_ADM_s105
                                 + 1.16192 w_k1_AXP_s105
                                 + 0.917219 w_k1_DIS_s105
                                 + 1.08185 w_k1_FDX_s105
                                 + 0.671768 w_k1_IBM_s105
                                 + 0.946389 w_k1_INTC_s105
                                 + 0.733314 w_k1_KO_s105
                                 + 0.810583 w_k1_MAT_s105
                                 + 1.09615 w_k1_MSFT_s105
                                 + 0.858162 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.982115 w_k1_ADM_s106
                                 + 1.00256 w_k1_AXP_s106
                                 + 1.05202 w_k1_DIS_s106
                                 + 1.06356 w_k1_FDX_s106
                                 + 0.923962 w_k1_IBM_s106
                                 + 0.927139 w_k1_INTC_s106
                                 + 0.888634 w_k1_KO_s106
                                 + 0.877318 w_k1_MAT_s106
                                 + 0.846234 w_k1_MSFT_s106
                                 + 1.01324 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.825441 w_k1_ADM_s106
                                 + 0.978672 w_k1_AXP_s106
                                 + 0.842336 w_k1_DIS_s106
                                 + 0.905921 w_k1_FDX_s106
                                 + 1.50036 w_k1_IBM_s106
                                 + 0.945631 w_k1_INTC_s106
                                 + 0.948472 w_k1_KO_s106
                                 + 1.08186 w_k1_MAT_s106
                                 + 0.860744 w_k1_MSFT_s106
                                 + 1.04408 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.22598 w_k1_ADM_s106
                                 + 1.41201 w_k1_AXP_s106
                                 + 1.58689 w_k1_DIS_s106
                                 + 1.39774 w_k1_FDX_s106
                                 + 1.14547 w_k1_IBM_s106
                                 + 1.41344 w_k1_INTC_s106
                                 + 1.36468 w_k1_KO_s106 + 1.04699 w_k1_MAT_s106
                                 + 1.28269 w_k1_MSFT_s106
                                 + 1.38042 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.924834 w_k1_ADM_s106
                                 + 0.719436 w_k1_AXP_s106
                                 + 1.08256 w_k1_DIS_s106
                                 + 0.982057 w_k1_FDX_s106
                                 + 1.0931 w_k1_IBM_s106
                                 + 1.03759 w_k1_INTC_s106
                                 + 0.93582 w_k1_KO_s106
                                 + 0.611678 w_k1_MAT_s106
                                 + 1.04586 w_k1_MSFT_s106
                                 + 1.03608 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.833618 w_k1_ADM_s106
                                 + 0.699945 w_k1_AXP_s106
                                 + 0.657306 w_k1_DIS_s106
                                 + 0.732514 w_k1_FDX_s106 + 1.068 w_k1_IBM_s106
                                 + 0.909726 w_k1_INTC_s106
                                 + 0.818669 w_k1_KO_s106
                                 + 0.612132 w_k1_MAT_s106
                                 + 0.751237 w_k1_MSFT_s106
                                 + 0.853661 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.808306 w_k1_ADM_s106
                                 + 0.7401 w_k1_AXP_s106
                                 + 0.746881 w_k1_DIS_s106
                                 + 0.861853 w_k1_FDX_s106
                                 + 0.616522 w_k1_IBM_s106
                                 + 0.851088 w_k1_INTC_s106
                                 + 0.686584 w_k1_KO_s106
                                 + 0.491784 w_k1_MAT_s106
                                 + 0.817799 w_k1_MSFT_s106
                                 + 0.829723 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.870916 w_k1_ADM_s106
                                 + 0.790139 w_k1_AXP_s106
                                 + 0.777853 w_k1_DIS_s106
                                 + 0.86474 w_k1_FDX_s106
                                 + 0.986863 w_k1_IBM_s106
                                 + 1.00097 w_k1_INTC_s106
                                 + 0.922416 w_k1_KO_s106
                                 + 1.14434 w_k1_MAT_s106
                                 + 0.891524 w_k1_MSFT_s106
                                 + 0.974494 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.689084 w_k1_ADM_s106
                                 + 0.574273 w_k1_AXP_s106
                                 + 0.554515 w_k1_DIS_s106
                                 + 0.700171 w_k1_FDX_s106
                                 + 0.704234 w_k1_IBM_s106
                                 + 0.915481 w_k1_INTC_s106
                                 + 0.680912 w_k1_KO_s106
                                 + 0.644401 w_k1_MAT_s106
                                 + 0.741874 w_k1_MSFT_s106
                                 + 0.87254 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.28529 w_k1_ADM_s107
                                 + 1.36684 w_k1_AXP_s107
                                 + 1.54307 w_k1_DIS_s107
                                 + 1.27514 w_k1_FDX_s107
                                 + 1.52868 w_k1_IBM_s107
                                 + 1.30756 w_k1_INTC_s107 + 1.6826 w_k1_KO_s107
                                 + 2.22374 w_k1_MAT_s107
                                 + 1.3606 w_k1_MSFT_s107 + 1.3718 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.942779 w_k1_ADM_s107
                                 + 0.692518 w_k1_AXP_s107
                                 + 0.710543 w_k1_DIS_s107
                                 + 0.892896 w_k1_FDX_s107
                                 + 0.664392 w_k1_IBM_s107
                                 + 0.813265 w_k1_INTC_s107
                                 + 0.735609 w_k1_KO_s107
                                 + 0.746397 w_k1_MAT_s107
                                 + 0.853409 w_k1_MSFT_s107
                                 + 0.895671 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.30633 w_k1_ADM_s107
                                 + 1.27196 w_k1_AXP_s107
                                 + 1.13133 w_k1_DIS_s107 + 1.178 w_k1_FDX_s107
                                 + 0.966431 w_k1_IBM_s107
                                 + 1.09834 w_k1_INTC_s107
                                 + 1.10335 w_k1_KO_s107 + 1.22536 w_k1_MAT_s107
                                 + 1.08214 w_k1_MSFT_s107
                                 + 1.05025 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.758875 w_k1_ADM_s107
                                 + 0.636862 w_k1_AXP_s107
                                 + 0.609469 w_k1_DIS_s107
                                 + 0.735943 w_k1_FDX_s107
                                 + 0.713711 w_k1_IBM_s107
                                 + 0.852107 w_k1_INTC_s107
                                 + 0.831509 w_k1_KO_s107
                                 + 0.790927 w_k1_MAT_s107
                                 + 0.870926 w_k1_MSFT_s107
                                 + 0.844209 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.983252 w_k1_ADM_s107
                                 + 0.866952 w_k1_AXP_s107
                                 + 0.757654 w_k1_DIS_s107
                                 + 0.916892 w_k1_FDX_s107
                                 + 1.15063 w_k1_IBM_s107
                                 + 0.932742 w_k1_INTC_s107
                                 + 1.04176 w_k1_KO_s107 + 1.34458 w_k1_MAT_s107
                                 + 0.819007 w_k1_MSFT_s107
                                 + 0.959136 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.02455 w_k1_ADM_s107
                                 + 0.881437 w_k1_AXP_s107
                                 + 0.807179 w_k1_DIS_s107
                                 + 0.957994 w_k1_FDX_s107
                                 + 0.970722 w_k1_IBM_s107
                                 + 0.969944 w_k1_INTC_s107
                                 + 0.949691 w_k1_KO_s107
                                 + 1.10717 w_k1_MAT_s107
                                 + 0.93298 w_k1_MSFT_s107
                                 + 0.993581 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.785811 w_k1_ADM_s107
                                 + 0.810025 w_k1_AXP_s107
                                 + 1.00539 w_k1_DIS_s107
                                 + 1.02215 w_k1_FDX_s107
                                 + 0.784699 w_k1_IBM_s107
                                 + 0.971629 w_k1_INTC_s107
                                 + 0.964181 w_k1_KO_s107
                                 + 0.991062 w_k1_MAT_s107
                                 + 1.16688 w_k1_MSFT_s107
                                 + 1.01582 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.977962 w_k1_ADM_s107
                                 + 0.926501 w_k1_AXP_s107
                                 + 1.09286 w_k1_DIS_s107
                                 + 1.06113 w_k1_FDX_s107 + 1.0858 w_k1_IBM_s107
                                 + 1.06303 w_k1_INTC_s107
                                 + 1.04811 w_k1_KO_s107 + 1.00538 w_k1_MAT_s107
                                 + 0.983809 w_k1_MSFT_s107
                                 + 1.08581 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.990142 w_k1_ADM_s110
                                 + 1.0901 w_k1_AXP_s110
                                 + 0.947484 w_k1_DIS_s110
                                 + 0.923903 w_k1_FDX_s110
                                 + 1.39996 w_k1_IBM_s110
                                 + 0.815179 w_k1_INTC_s110
                                 + 1.09535 w_k1_KO_s110 + 1.24897 w_k1_MAT_s110
                                 + 0.845653 w_k1_MSFT_s110
                                 + 1.01554 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.11298 w_k1_ADM_s110
                                 + 1.28652 w_k1_AXP_s110
                                 + 1.31983 w_k1_DIS_s110
                                 + 1.17062 w_k1_FDX_s110
                                 + 1.14195 w_k1_IBM_s110
                                 + 1.12002 w_k1_INTC_s110
                                 + 1.16615 w_k1_KO_s110 + 1.30861 w_k1_MAT_s110
                                 + 1.14838 w_k1_MSFT_s110
                                 + 1.11101 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.792837 w_k1_ADM_s110
                                 + 0.696431 w_k1_AXP_s110
                                 + 0.550776 w_k1_DIS_s110
                                 + 0.80558 w_k1_FDX_s110
                                 + 0.695808 w_k1_IBM_s110
                                 + 0.752151 w_k1_INTC_s110
                                 + 0.619678 w_k1_KO_s110
                                 + 0.642992 w_k1_MAT_s110
                                 + 0.700496 w_k1_MSFT_s110
                                 + 0.884139 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.13006 w_k1_ADM_s110
                                 + 1.18612 w_k1_AXP_s110
                                 + 0.941144 w_k1_DIS_s110
                                 + 1.08007 w_k1_FDX_s110
                                 + 0.812668 w_k1_IBM_s110
                                 + 0.916948 w_k1_INTC_s110
                                 + 0.831609 w_k1_KO_s110
                                 + 0.986869 w_k1_MAT_s110
                                 + 0.942141 w_k1_MSFT_s110
                                 + 1.03005 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.931783 w_k1_ADM_s110
                                 + 0.956736 w_k1_AXP_s110
                                 + 0.823726 w_k1_DIS_s110
                                 + 0.821381 w_k1_FDX_s110
                                 + 0.982252 w_k1_IBM_s110
                                 + 0.860883 w_k1_INTC_s110
                                 + 0.85349 w_k1_KO_s110
                                 + 0.994617 w_k1_MAT_s110
                                 + 0.78924 w_k1_MSFT_s110
                                 + 1.02057 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.00108 w_k1_ADM_s110
                                 + 0.97041 w_k1_AXP_s110
                                 + 0.750492 w_k1_DIS_s110
                                 + 0.884433 w_k1_FDX_s110
                                 + 0.716222 w_k1_IBM_s110
                                 + 0.928227 w_k1_INTC_s110
                                 + 0.666545 w_k1_KO_s110
                                 + 0.477181 w_k1_MAT_s110
                                 + 0.853344 w_k1_MSFT_s110
                                 + 0.920937 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.14014 w_k1_ADM_s110
                                 + 1.36439 w_k1_AXP_s110
                                 + 1.67023 w_k1_DIS_s110
                                 + 1.44012 w_k1_FDX_s110
                                 + 1.54317 w_k1_IBM_s110
                                 + 1.32527 w_k1_INTC_s110
                                 + 1.32774 w_k1_KO_s110 + 1.09374 w_k1_MAT_s110
                                 + 1.23681 w_k1_MSFT_s110
                                 + 1.30338 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.26235 w_k1_ADM_s110
                                 + 1.00868 w_k1_AXP_s110
                                 + 1.05411 w_k1_DIS_s110
                                 + 1.11333 w_k1_FDX_s110
                                 + 0.878459 w_k1_IBM_s110
                                 + 1.06899 w_k1_INTC_s110
                                 + 0.853921 w_k1_KO_s110
                                 + 0.547033 w_k1_MAT_s110
                                 + 0.991633 w_k1_MSFT_s110
                                 + 1.01786 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.907639 w_k1_ADM_s111
                                 + 1.00023 w_k1_AXP_s111
                                 + 0.728065 w_k1_DIS_s111
                                 + 0.955146 w_k1_FDX_s111
                                 + 0.77134 w_k1_IBM_s111
                                 + 0.837452 w_k1_INTC_s111
                                 + 0.752898 w_k1_KO_s111
                                 + 1.08806 w_k1_MAT_s111
                                 + 0.855434 w_k1_MSFT_s111
                                 + 1.03744 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.05095 w_k1_ADM_s111
                                 + 0.953765 w_k1_AXP_s111
                                 + 1.03505 w_k1_DIS_s111
                                 + 1.20976 w_k1_FDX_s111
                                 + 0.671978 w_k1_IBM_s111
                                 + 1.14849 w_k1_INTC_s111
                                 + 0.888145 w_k1_KO_s111
                                 + 0.827872 w_k1_MAT_s111
                                 + 1.20147 w_k1_MSFT_s111
                                 + 0.969702 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.01975 w_k1_ADM_s111
                                 + 1.00656 w_k1_AXP_s111
                                 + 1.28481 w_k1_DIS_s111
                                 + 1.12059 w_k1_FDX_s111
                                 + 1.37542 w_k1_IBM_s111
                                 + 1.17809 w_k1_INTC_s111
                                 + 1.16468 w_k1_KO_s111 + 1.25948 w_k1_MAT_s111
                                 + 1.04058 w_k1_MSFT_s111
                                 + 1.03099 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.925394 w_k1_ADM_s111
                                 + 0.775658 w_k1_AXP_s111
                                 + 0.821993 w_k1_DIS_s111
                                 + 0.872348 w_k1_FDX_s111
                                 + 0.880898 w_k1_IBM_s111
                                 + 1.03602 w_k1_INTC_s111
                                 + 0.806387 w_k1_KO_s111
                                 + 0.834493 w_k1_MAT_s111
                                 + 0.894308 w_k1_MSFT_s111
                                 + 0.911703 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.884528 w_k1_ADM_s111
                                 + 0.748979 w_k1_AXP_s111
                                 + 0.594521 w_k1_DIS_s111
                                 + 0.795747 w_k1_FDX_s111
                                 + 0.608768 w_k1_IBM_s111
                                 + 0.883949 w_k1_INTC_s111
                                 + 0.662479 w_k1_KO_s111
                                 + 0.550477 w_k1_MAT_s111
                                 + 0.754956 w_k1_MSFT_s111
                                 + 0.8039 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.804066 w_k1_ADM_s111
                                 + 0.950311 w_k1_AXP_s111
                                 + 0.618372 w_k1_DIS_s111
                                 + 0.779367 w_k1_FDX_s111
                                 + 0.942261 w_k1_IBM_s111
                                 + 0.759153 w_k1_INTC_s111
                                 + 0.677021 w_k1_KO_s111
                                 + 1.36626 w_k1_MAT_s111
                                 + 0.715569 w_k1_MSFT_s111
                                 + 0.841171 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.1258 w_k1_ADM_s111
                                 + 1.41982 w_k1_AXP_s111
                                 + 1.15454 w_k1_DIS_s111
                                 + 1.22102 w_k1_FDX_s111 + 1.0234 w_k1_IBM_s111
                                 + 1.44331 w_k1_INTC_s111
                                 + 1.02966 w_k1_KO_s111 + 1.2864 w_k1_MAT_s111
                                 + 1.06053 w_k1_MSFT_s111
                                 + 1.12673 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.823074 w_k1_ADM_s111
                                 + 1.29146 w_k1_AXP_s111 + 0.9844 w_k1_DIS_s111
                                 + 1.02798 w_k1_FDX_s111
                                 + 1.20305 w_k1_IBM_s111
                                 + 1.17725 w_k1_INTC_s111
                                 + 1.08927 w_k1_KO_s111 + 1.58655 w_k1_MAT_s111
                                 + 1.06014 w_k1_MSFT_s111
                                 + 1.03847 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.858556 w_k1_ADM_s112
                                 + 0.763725 w_k1_AXP_s112
                                 + 1.17194 w_k1_DIS_s112
                                 + 0.956744 w_k1_FDX_s112
                                 + 1.09035 w_k1_IBM_s112
                                 + 0.957878 w_k1_INTC_s112
                                 + 1.05789 w_k1_KO_s112 + 0.99668 w_k1_MAT_s112
                                 + 1.06766 w_k1_MSFT_s112
                                 + 0.96064 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.750186 w_k1_ADM_s112
                                 + 0.787589 w_k1_AXP_s112
                                 + 0.682537 w_k1_DIS_s112
                                 + 0.709842 w_k1_FDX_s112
                                 + 0.982012 w_k1_IBM_s112
                                 + 0.771527 w_k1_INTC_s112
                                 + 0.786977 w_k1_KO_s112
                                 + 0.810944 w_k1_MAT_s112
                                 + 0.79411 w_k1_MSFT_s112
                                 + 0.791289 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.711811 w_k1_ADM_s112
                                 + 0.835697 w_k1_AXP_s112
                                 + 0.745612 w_k1_DIS_s112
                                 + 0.910887 w_k1_FDX_s112
                                 + 0.660602 w_k1_IBM_s112
                                 + 0.906098 w_k1_INTC_s112
                                 + 0.664119 w_k1_KO_s112
                                 + 0.717446 w_k1_MAT_s112
                                 + 0.846599 w_k1_MSFT_s112
                                 + 0.86159 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.14148 w_k1_ADM_s112
                                 + 1.3903 w_k1_AXP_s112 + 1.43194 w_k1_DIS_s112
                                 + 1.19593 w_k1_FDX_s112
                                 + 1.49794 w_k1_IBM_s112
                                 + 1.09022 w_k1_INTC_s112
                                 + 1.40039 w_k1_KO_s112 + 1.39098 w_k1_MAT_s112
                                 + 1.2102 w_k1_MSFT_s112 + 1.15896 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.056 w_k1_ADM_s112
                                 + 1.03052 w_k1_AXP_s112
                                 + 0.775342 w_k1_DIS_s112
                                 + 0.903086 w_k1_FDX_s112
                                 + 0.806591 w_k1_IBM_s112
                                 + 0.783391 w_k1_INTC_s112
                                 + 0.800164 w_k1_KO_s112
                                 + 1.00924 w_k1_MAT_s112
                                 + 0.866931 w_k1_MSFT_s112
                                 + 0.822585 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.97147 w_k1_ADM_s112
                                 + 1.12767 w_k1_AXP_s112
                                 + 1.08076 w_k1_DIS_s112
                                 + 1.16388 w_k1_FDX_s112
                                 + 0.675486 w_k1_IBM_s112
                                 + 1.07518 w_k1_INTC_s112
                                 + 0.889046 w_k1_KO_s112
                                 + 0.888602 w_k1_MAT_s112
                                 + 1.13981 w_k1_MSFT_s112
                                 + 0.94131 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.08335 w_k1_ADM_s112
                                 + 1.19842 w_k1_AXP_s112 + 1.097 w_k1_DIS_s112
                                 + 1.16389 w_k1_FDX_s112
                                 + 0.870785 w_k1_IBM_s112
                                 + 1.07299 w_k1_INTC_s112
                                 + 1.04912 w_k1_KO_s112 + 1.15902 w_k1_MAT_s112
                                 + 1.02842 w_k1_MSFT_s112
                                 + 1.08965 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.934566 w_k1_ADM_s112
                                 + 1.27839 w_k1_AXP_s112 + 1.0318 w_k1_DIS_s112
                                 + 1.02066 w_k1_FDX_s112
                                 + 1.18471 w_k1_IBM_s112
                                 + 1.15288 w_k1_INTC_s112
                                 + 1.14958 w_k1_KO_s112 + 1.44101 w_k1_MAT_s112
                                 + 1.01405 w_k1_MSFT_s112 + 1.016 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.780938 w_k1_ADM_s113
                                 + 0.757565 w_k1_AXP_s113
                                 + 1.04931 w_k1_DIS_s113
                                 + 0.814789 w_k1_FDX_s113
                                 + 1.3932 w_k1_IBM_s113
                                 + 0.995506 w_k1_INTC_s113
                                 + 1.17098 w_k1_KO_s113 + 1.22188 w_k1_MAT_s113
                                 + 0.941905 w_k1_MSFT_s113
                                 + 0.946257 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.767961 w_k1_ADM_s113
                                 + 0.753096 w_k1_AXP_s113
                                 + 0.674951 w_k1_DIS_s113
                                 + 0.783201 w_k1_FDX_s113
                                 + 0.546806 w_k1_IBM_s113
                                 + 0.872659 w_k1_INTC_s113
                                 + 0.710851 w_k1_KO_s113
                                 + 0.581323 w_k1_MAT_s113
                                 + 0.849286 w_k1_MSFT_s113
                                 + 0.754935 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.17443 w_k1_ADM_s113
                                 + 0.997146 w_k1_AXP_s113
                                 + 0.93145 w_k1_DIS_s113
                                 + 1.12344 w_k1_FDX_s113
                                 + 0.678323 w_k1_IBM_s113
                                 + 1.02098 w_k1_INTC_s113
                                 + 0.817481 w_k1_KO_s113
                                 + 0.750308 w_k1_MAT_s113
                                 + 0.988527 w_k1_MSFT_s113
                                 + 0.924897 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11407 w_k1_ADM_s113
                                 + 1.1189 w_k1_AXP_s113 + 1.29642 w_k1_DIS_s113
                                 + 1.12026 w_k1_FDX_s113
                                 + 1.29399 w_k1_IBM_s113
                                 + 1.01456 w_k1_INTC_s113 + 1.3247 w_k1_KO_s113
                                 + 1.24821 w_k1_MAT_s113
                                 + 1.02514 w_k1_MSFT_s113
                                 + 1.12598 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.98059 w_k1_ADM_s113
                                 + 1.1784 w_k1_AXP_s113 + 1.5714 w_k1_DIS_s113
                                 + 1.13983 w_k1_FDX_s113
                                 + 1.90952 w_k1_IBM_s113
                                 + 1.26536 w_k1_INTC_s113
                                 + 1.63666 w_k1_KO_s113 + 1.72707 w_k1_MAT_s113
                                 + 1.04789 w_k1_MSFT_s113 + 1.1836 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.16526 w_k1_ADM_s113
                                 + 1.25499 w_k1_AXP_s113
                                 + 1.54242 w_k1_DIS_s113 + 1.3621 w_k1_FDX_s113
                                 + 1.08463 w_k1_IBM_s113
                                 + 1.26516 w_k1_INTC_s113
                                 + 1.37221 w_k1_KO_s113 + 1.13008 w_k1_MAT_s113
                                 + 1.21651 w_k1_MSFT_s113
                                 + 1.00289 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.12421 w_k1_ADM_s113
                                 + 1.15735 w_k1_AXP_s113
                                 + 1.12092 w_k1_DIS_s113
                                 + 1.04226 w_k1_FDX_s113
                                 + 1.02613 w_k1_IBM_s113
                                 + 1.16624 w_k1_INTC_s113
                                 + 0.991569 w_k1_KO_s113
                                 + 0.749221 w_k1_MAT_s113
                                 + 1.00762 w_k1_MSFT_s113
                                 + 1.07555 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.880628 w_k1_ADM_s113
                                 + 0.877054 w_k1_AXP_s113
                                 + 0.797729 w_k1_DIS_s113
                                 + 0.859236 w_k1_FDX_s113
                                 + 0.955389 w_k1_IBM_s113
                                 + 0.9301 w_k1_INTC_s113
                                 + 0.829421 w_k1_KO_s113
                                 + 0.570708 w_k1_MAT_s113
                                 + 0.806108 w_k1_MSFT_s113
                                 + 0.87375 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.889557 w_k1_ADM_s114
                                 + 0.716729 w_k1_AXP_s114
                                 + 0.723682 w_k1_DIS_s114
                                 + 0.827124 w_k1_FDX_s114
                                 + 0.645761 w_k1_IBM_s114
                                 + 0.947654 w_k1_INTC_s114
                                 + 0.680827 w_k1_KO_s114
                                 + 0.680112 w_k1_MAT_s114
                                 + 0.834687 w_k1_MSFT_s114
                                 + 0.752652 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.33623 w_k1_ADM_s114
                                 + 1.25155 w_k1_AXP_s114
                                 + 1.62486 w_k1_DIS_s114
                                 + 1.33863 w_k1_FDX_s114
                                 + 1.12442 w_k1_IBM_s114 + 1.199 w_k1_INTC_s114
                                 + 1.14849 w_k1_KO_s114 + 1.35404 w_k1_MAT_s114
                                 + 1.31966 w_k1_MSFT_s114
                                 + 1.23883 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.907443 w_k1_ADM_s114
                                 + 0.735893 w_k1_AXP_s114
                                 + 0.858823 w_k1_DIS_s114
                                 + 0.826978 w_k1_FDX_s114
                                 + 1.05842 w_k1_IBM_s114
                                 + 0.956067 w_k1_INTC_s114
                                 + 0.891345 w_k1_KO_s114
                                 + 0.861143 w_k1_MAT_s114
                                 + 0.812986 w_k1_MSFT_s114
                                 + 0.921787 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.93207 w_k1_ADM_s114
                                 + 1.07276 w_k1_AXP_s114
                                 + 1.41063 w_k1_DIS_s114
                                 + 1.05531 w_k1_FDX_s114
                                 + 1.56202 w_k1_IBM_s114
                                 + 1.42863 w_k1_INTC_s114
                                 + 1.27926 w_k1_KO_s114 + 1.45465 w_k1_MAT_s114
                                 + 1.15378 w_k1_MSFT_s114
                                 + 1.16088 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.1756 w_k1_ADM_s114
                                 + 1.06967 w_k1_AXP_s114
                                 + 1.12684 w_k1_DIS_s114 + 1.1928 w_k1_FDX_s114
                                 + 0.803803 w_k1_IBM_s114
                                 + 0.96528 w_k1_INTC_s114
                                 + 0.86887 w_k1_KO_s114
                                 + 0.823903 w_k1_MAT_s114
                                 + 1.15835 w_k1_MSFT_s114
                                 + 1.01898 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.14191 w_k1_ADM_s114
                                 + 1.37171 w_k1_AXP_s114
                                 + 0.958903 w_k1_DIS_s114
                                 + 1.15609 w_k1_FDX_s114
                                 + 0.743851 w_k1_IBM_s114
                                 + 0.94985 w_k1_INTC_s114
                                 + 0.629441 w_k1_KO_s114
                                 + 0.923772 w_k1_MAT_s114
                                 + 0.948058 w_k1_MSFT_s114
                                 + 0.926869 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.55777 w_k1_ADM_s114
                                 + 1.69236 w_k1_AXP_s114
                                 + 1.82775 w_k1_DIS_s114
                                 + 1.48805 w_k1_FDX_s114
                                 + 1.54365 w_k1_IBM_s114
                                 + 1.30011 w_k1_INTC_s114
                                 + 1.37339 w_k1_KO_s114 + 1.5656 w_k1_MAT_s114
                                 + 1.29624 w_k1_MSFT_s114
                                 + 1.16524 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.891377 w_k1_ADM_s114
                                 + 0.910536 w_k1_AXP_s114
                                 + 1.15409 w_k1_DIS_s114
                                 + 1.03186 w_k1_FDX_s114
                                 + 0.987278 w_k1_IBM_s114
                                 + 1.05101 w_k1_INTC_s114
                                 + 0.93455 w_k1_KO_s114 + 1.09959 w_k1_MAT_s114
                                 + 1.06432 w_k1_MSFT_s114
                                 + 1.01687 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.23939 w_k1_ADM_s115
                                 + 1.50493 w_k1_AXP_s115
                                 + 1.87696 w_k1_DIS_s115
                                 + 1.49998 w_k1_FDX_s115
                                 + 1.61971 w_k1_IBM_s115
                                 + 1.77204 w_k1_INTC_s115
                                 + 1.79416 w_k1_KO_s115 + 1.29464 w_k1_MAT_s115
                                 + 1.45518 w_k1_MSFT_s115
                                 + 1.20526 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.936535 w_k1_ADM_s115
                                 + 1.18478 w_k1_AXP_s115
                                 + 0.819888 w_k1_DIS_s115
                                 + 0.965119 w_k1_FDX_s115
                                 + 1.13315 w_k1_IBM_s115
                                 + 1.26033 w_k1_INTC_s115
                                 + 0.971048 w_k1_KO_s115
                                 + 1.14996 w_k1_MAT_s115
                                 + 0.944848 w_k1_MSFT_s115
                                 + 0.926354 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.975512 w_k1_ADM_s115
                                 + 1.15617 w_k1_AXP_s115
                                 + 1.50247 w_k1_DIS_s115
                                 + 1.21495 w_k1_FDX_s115
                                 + 1.49375 w_k1_IBM_s115
                                 + 1.26035 w_k1_INTC_s115
                                 + 1.37063 w_k1_KO_s115 + 1.8085 w_k1_MAT_s115
                                 + 1.24125 w_k1_MSFT_s115
                                 + 1.27627 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01436 w_k1_ADM_s115
                                 + 1.02243 w_k1_AXP_s115
                                 + 1.20354 w_k1_DIS_s115
                                 + 0.96739 w_k1_FDX_s115
                                 + 1.41455 w_k1_IBM_s115
                                 + 1.28148 w_k1_INTC_s115
                                 + 1.33454 w_k1_KO_s115 + 1.38993 w_k1_MAT_s115
                                 + 1.15013 w_k1_MSFT_s115
                                 + 0.989315 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.805644 w_k1_ADM_s115
                                 + 0.652011 w_k1_AXP_s115
                                 + 0.571964 w_k1_DIS_s115
                                 + 0.777355 w_k1_FDX_s115
                                 + 0.594687 w_k1_IBM_s115
                                 + 0.809757 w_k1_INTC_s115
                                 + 0.661041 w_k1_KO_s115
                                 + 0.604823 w_k1_MAT_s115
                                 + 0.733744 w_k1_MSFT_s115
                                 + 0.850975 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.856733 w_k1_ADM_s115
                                 + 0.981239 w_k1_AXP_s115
                                 + 1.12862 w_k1_DIS_s115
                                 + 1.04838 w_k1_FDX_s115
                                 + 1.08276 w_k1_IBM_s115
                                 + 1.16696 w_k1_INTC_s115
                                 + 1.24715 w_k1_KO_s115
                                 + 0.942737 w_k1_MAT_s115
                                 + 1.09319 w_k1_MSFT_s115
                                 + 1.04785 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.923926 w_k1_ADM_s115
                                 + 0.892073 w_k1_AXP_s115
                                 + 1.03171 w_k1_DIS_s115 + 1.206 w_k1_FDX_s115
                                 + 0.664496 w_k1_IBM_s115
                                 + 1.16294 w_k1_INTC_s115
                                 + 0.892416 w_k1_KO_s115
                                 + 0.622745 w_k1_MAT_s115
                                 + 1.2756 w_k1_MSFT_s115 + 0.95212 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.908836 w_k1_ADM_s115
                                 + 0.700541 w_k1_AXP_s115
                                 + 0.752333 w_k1_DIS_s115
                                 + 0.8098 w_k1_FDX_s115
                                 + 0.884129 w_k1_IBM_s115
                                 + 1.00483 w_k1_INTC_s115
                                 + 0.909367 w_k1_KO_s115
                                 + 0.55202 w_k1_MAT_s115
                                 + 0.850961 w_k1_MSFT_s115
                                 + 0.851103 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.691277 w_k1_ADM_s116
                                 + 0.956574 w_k1_AXP_s116
                                 + 0.642325 w_k1_DIS_s116
                                 + 0.81039 w_k1_FDX_s116
                                 + 1.39818 w_k1_IBM_s116
                                 + 1.13924 w_k1_INTC_s116
                                 + 0.980591 w_k1_KO_s116
                                 + 1.07643 w_k1_MAT_s116
                                 + 0.762646 w_k1_MSFT_s116
                                 + 0.838357 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.11642 w_k1_ADM_s116
                                 + 1.54147 w_k1_AXP_s116
                                 + 1.76603 w_k1_DIS_s116
                                 + 1.34249 w_k1_FDX_s116
                                 + 1.19172 w_k1_IBM_s116
                                 + 1.8627 w_k1_INTC_s116 + 1.55501 w_k1_KO_s116
                                 + 1.13446 w_k1_MAT_s116
                                 + 1.3895 w_k1_MSFT_s116 + 1.22876 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.997514 w_k1_ADM_s116
                                 + 1.41538 w_k1_AXP_s116
                                 + 1.56031 w_k1_DIS_s116
                                 + 1.20645 w_k1_FDX_s116
                                 + 1.84501 w_k1_IBM_s116
                                 + 1.21792 w_k1_INTC_s116
                                 + 1.62168 w_k1_KO_s116 + 2.15622 w_k1_MAT_s116
                                 + 1.36514 w_k1_MSFT_s116
                                 + 1.05184 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.31388 w_k1_ADM_s116
                                 + 1.21352 w_k1_AXP_s116
                                 + 1.02714 w_k1_DIS_s116 + 1.0103 w_k1_FDX_s116
                                 + 1.32493 w_k1_IBM_s116
                                 + 0.971812 w_k1_INTC_s116
                                 + 1.13917 w_k1_KO_s116 + 1.51005 w_k1_MAT_s116
                                 + 0.905942 w_k1_MSFT_s116
                                 + 1.04179 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.845353 w_k1_ADM_s116
                                 + 0.923143 w_k1_AXP_s116
                                 + 0.763375 w_k1_DIS_s116
                                 + 0.823099 w_k1_FDX_s116
                                 + 0.999646 w_k1_IBM_s116
                                 + 0.905871 w_k1_INTC_s116
                                 + 0.896076 w_k1_KO_s116
                                 + 0.95337 w_k1_MAT_s116
                                 + 0.895006 w_k1_MSFT_s116
                                 + 0.761614 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.12877 w_k1_ADM_s116
                                 + 1.43587 w_k1_AXP_s116
                                 + 1.07727 w_k1_DIS_s116
                                 + 1.05652 w_k1_FDX_s116
                                 + 0.935208 w_k1_IBM_s116
                                 + 1.1219 w_k1_INTC_s116 + 1.10138 w_k1_KO_s116
                                 + 0.931747 w_k1_MAT_s116
                                 + 1.00871 w_k1_MSFT_s116
                                 + 0.901504 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.968674 w_k1_ADM_s116
                                 + 0.858794 w_k1_AXP_s116
                                 + 0.673234 w_k1_DIS_s116
                                 + 0.811742 w_k1_FDX_s116
                                 + 0.698732 w_k1_IBM_s116
                                 + 0.958282 w_k1_INTC_s116
                                 + 0.771072 w_k1_KO_s116
                                 + 0.722239 w_k1_MAT_s116
                                 + 0.783776 w_k1_MSFT_s116
                                 + 0.683459 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.940211 w_k1_ADM_s116
                                 + 1.13184 w_k1_AXP_s116
                                 + 1.01988 w_k1_DIS_s116 + 1.1573 w_k1_FDX_s116
                                 + 0.82245 w_k1_IBM_s116
                                 + 1.19948 w_k1_INTC_s116
                                 + 0.931559 w_k1_KO_s116
                                 + 0.796523 w_k1_MAT_s116
                                 + 0.994005 w_k1_MSFT_s116
                                 + 0.932614 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.4043 w_k1_ADM_s117
                                 + 0.980698 w_k1_AXP_s117
                                 + 1.00203 w_k1_DIS_s117
                                 + 0.950887 w_k1_FDX_s117
                                 + 1.11618 w_k1_IBM_s117
                                 + 1.14438 w_k1_INTC_s117
                                 + 1.09596 w_k1_KO_s117
                                 + 0.768046 w_k1_MAT_s117
                                 + 1.00087 w_k1_MSFT_s117
                                 + 0.994805 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.11442 w_k1_ADM_s117
                                 + 0.846571 w_k1_AXP_s117
                                 + 0.669443 w_k1_DIS_s117
                                 + 0.933804 w_k1_FDX_s117
                                 + 0.79259 w_k1_IBM_s117
                                 + 1.06235 w_k1_INTC_s117
                                 + 0.79023 w_k1_KO_s117
                                 + 0.701424 w_k1_MAT_s117
                                 + 0.85452 w_k1_MSFT_s117
                                 + 1.03302 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.5456 w_k1_ADM_s117
                                 + 1.41298 w_k1_AXP_s117 + 1.4906 w_k1_DIS_s117
                                 + 1.25091 w_k1_FDX_s117
                                 + 1.40434 w_k1_IBM_s117
                                 + 1.40689 w_k1_INTC_s117
                                 + 1.17183 w_k1_KO_s117 + 1.0572 w_k1_MAT_s117
                                 + 1.11212 w_k1_MSFT_s117
                                 + 1.19588 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.41446 w_k1_ADM_s117
                                 + 1.46314 w_k1_AXP_s117
                                 + 1.14704 w_k1_DIS_s117
                                 + 1.25953 w_k1_FDX_s117
                                 + 1.05685 w_k1_IBM_s117
                                 + 1.25392 w_k1_INTC_s117 + 1.0226 w_k1_KO_s117
                                 + 1.30175 w_k1_MAT_s117
                                 + 1.07972 w_k1_MSFT_s117
                                 + 1.23404 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.912243 w_k1_ADM_s117
                                 + 0.701473 w_k1_AXP_s117
                                 + 0.644874 w_k1_DIS_s117
                                 + 0.762508 w_k1_FDX_s117
                                 + 0.731379 w_k1_IBM_s117
                                 + 0.934426 w_k1_INTC_s117
                                 + 0.78957 w_k1_KO_s117
                                 + 0.811485 w_k1_MAT_s117
                                 + 0.793661 w_k1_MSFT_s117
                                 + 0.79713 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.11852 w_k1_ADM_s117
                                 + 0.938044 w_k1_AXP_s117
                                 + 0.742848 w_k1_DIS_s117
                                 + 0.877579 w_k1_FDX_s117
                                 + 0.88923 w_k1_IBM_s117
                                 + 0.930575 w_k1_INTC_s117
                                 + 0.818633 w_k1_KO_s117
                                 + 0.900841 w_k1_MAT_s117
                                 + 0.859155 w_k1_MSFT_s117
                                 + 0.973317 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.17907 w_k1_ADM_s117
                                 + 1.15668 w_k1_AXP_s117
                                 + 0.899336 w_k1_DIS_s117
                                 + 1.00318 w_k1_FDX_s117
                                 + 1.04008 w_k1_IBM_s117
                                 + 1.12957 w_k1_INTC_s117
                                 + 0.941573 w_k1_KO_s117
                                 + 0.936819 w_k1_MAT_s117
                                 + 0.975511 w_k1_MSFT_s117
                                 + 0.886521 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.51732 w_k1_ADM_s117
                                 + 0.912211 w_k1_AXP_s117
                                 + 1.35353 w_k1_DIS_s117
                                 + 1.19975 w_k1_FDX_s117
                                 + 0.786309 w_k1_IBM_s117
                                 + 1.38561 w_k1_INTC_s117
                                 + 0.941655 w_k1_KO_s117
                                 + 0.610368 w_k1_MAT_s117
                                 + 1.1882 w_k1_MSFT_s117 + 1.06123 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0198851590965632 w_k1_ADM_s000
                                 + 0.0162246805378536 w_k1_ADM_s001
                                 + 0.035018033497469 w_k1_ADM_s002
                                 + 0.0215063082785772 w_k1_ADM_s003
                                 + 0.064463251759321 w_k1_ADM_s004
                                 + 0.0273653714065191 w_k1_ADM_s005
                                 + 0.0163305905848689 w_k1_ADM_s006
                                 + 0.0134674775312759 w_k1_ADM_s007
                                 + 0.033350528347976 w_k1_ADM_s010
                                 + 0.0554997977126383 w_k1_ADM_s011
                                 + 0.0289953485890605 w_k1_ADM_s012
                                 + 0.0255525787437465 w_k1_ADM_s013
                                 + 0.029298449350136 w_k1_ADM_s014
                                 + 0.00994857706961688 w_k1_ADM_s015
                                 + 0.0254485928483272 w_k1_ADM_s016
                                 + 0.0374979306720947 w_k1_ADM_s017
                                 + 0.0179352393813295 w_k1_ADM_s100
                                 + 0.050257206314594 w_k1_ADM_s101
                                 + 0.0345631678756504 w_k1_ADM_s102
                                 + 0.0640177142455019 w_k1_ADM_s103
                                 + 0.0229372852289314 w_k1_ADM_s104
                                 + 0.0216091482132891 w_k1_ADM_s105
                                 + 0.0355375787139885 w_k1_ADM_s106
                                 + 0.0127669038763972 w_k1_ADM_s107
                                 + 0.0231436282551791 w_k1_ADM_s110
                                 + 0.022978332551614 w_k1_ADM_s111
                                 + 0.0222384301029843 w_k1_ADM_s112
                                 + 0.100150111176049 w_k1_ADM_s113
                                 + 0.085218804184785 w_k1_ADM_s114
                                 + 0.021953787155896 w_k1_ADM_s115
                                 + 0.0224096390265004 w_k1_ADM_s116
                                 + 0.0291177738540168 w_k1_ADM_s117
                                 + 0.0185297962825676 w_k1_AXP_s000
                                 + 0.0154369793290919 w_k1_AXP_s001
                                 + 0.0341540191426817 w_k1_AXP_s002
                                 + 0.0183183141845138 w_k1_AXP_s003
                                 + 0.0673879129339218 w_k1_AXP_s004
                                 + 0.0266918632839077 w_k1_AXP_s005
                                 + 0.0182162856487906 w_k1_AXP_s006
                                 + 0.0137215051012283 w_k1_AXP_s007
                                 + 0.0376914195794047 w_k1_AXP_s010
                                 + 0.0491724991168185 w_k1_AXP_s011
                                 + 0.0300083488807712 w_k1_AXP_s012
                                 + 0.0241377852230835 w_k1_AXP_s013
                                 + 0.0282767660473345 w_k1_AXP_s014
                                 + 0.0106446373576252 w_k1_AXP_s015
                                 + 0.0262415958442545 w_k1_AXP_s016
                                 + 0.0397837456733427 w_k1_AXP_s017
                                 + 0.016677967480895 w_k1_AXP_s100
                                 + 0.0515592158863081 w_k1_AXP_s101
                                 + 0.0324626620943462 w_k1_AXP_s102
                                 + 0.0458797794098062 w_k1_AXP_s103
                                 + 0.021512919593406 w_k1_AXP_s104
                                 + 0.0261279224305106 w_k1_AXP_s105
                                 + 0.0356833024147735 w_k1_AXP_s106
                                 + 0.0118429083652688 w_k1_AXP_s107
                                 + 0.0236891884326226 w_k1_AXP_s110
                                 + 0.025314254725818 w_k1_AXP_s111
                                 + 0.0248763822411685 w_k1_AXP_s112
                                 + 0.101068029492339 w_k1_AXP_s113
                                 + 0.0818410883646702 w_k1_AXP_s114
                                 + 0.022880476738904 w_k1_AXP_s115
                                 + 0.0260605564583291 w_k1_AXP_s116
                                 + 0.024557577251517 w_k1_AXP_s117
                                 + 0.0178683122805488 w_k1_DIS_s000
                                 + 0.0172862796663468 w_k1_DIS_s001
                                 + 0.0335302678646654 w_k1_DIS_s002
                                 + 0.0180800889062793 w_k1_DIS_s003
                                 + 0.0667061948999661 w_k1_DIS_s004
                                 + 0.0258061234603931 w_k1_DIS_s005
                                 + 0.0192806772474791 w_k1_DIS_s006
                                 + 0.0152243369712266 w_k1_DIS_s007
                                 + 0.0386185796118688 w_k1_DIS_s010
                                 + 0.0474601161486016 w_k1_DIS_s011
                                 + 0.0308396747709235 w_k1_DIS_s012
                                 + 0.0247082288532504 w_k1_DIS_s013
                                 + 0.0297898257117205 w_k1_DIS_s014
                                 + 0.0103952007348661 w_k1_DIS_s015
                                 + 0.0254172317076772 w_k1_DIS_s016
                                 + 0.0366387221673323 w_k1_DIS_s017
                                 + 0.0187309924951008 w_k1_DIS_s100
                                 + 0.0476257320909091 w_k1_DIS_s101
                                 + 0.0286088612233413 w_k1_DIS_s102
                                 + 0.0520790230471784 w_k1_DIS_s103
                                 + 0.0223076349048703 w_k1_DIS_s104
                                 + 0.0235280489870832 w_k1_DIS_s105
                                 + 0.037524800509229 w_k1_DIS_s106
                                 + 0.0116638802056226 w_k1_DIS_s107
                                 + 0.0220712307543123 w_k1_DIS_s110
                                 + 0.0229468880270414 w_k1_DIS_s111
                                 + 0.02511094538763 w_k1_DIS_s112
                                 + 0.108379723127602 w_k1_DIS_s113
                                 + 0.0931103845253939 w_k1_DIS_s114
                                 + 0.0246123348678377 w_k1_DIS_s115
                                 + 0.0233859811825799 w_k1_DIS_s116
                                 + 0.0232270076191354 w_k1_DIS_s117
                                 + 0.0185576463739543 w_k1_FDX_s000
                                 + 0.016680715327211 w_k1_FDX_s001
                                 + 0.0332031519542011 w_k1_FDX_s002
                                 + 0.0190822180993389 w_k1_FDX_s003
                                 + 0.0670722480599636 w_k1_FDX_s004
                                 + 0.0248173933769659 w_k1_FDX_s005
                                 + 0.0174374143445662 w_k1_FDX_s006
                                 + 0.014141860200374 w_k1_FDX_s007
                                 + 0.0369695908481877 w_k1_FDX_s010
                                 + 0.0489152016968678 w_k1_FDX_s011
                                 + 0.0292863736641598 w_k1_FDX_s012
                                 + 0.022347619965872 w_k1_FDX_s013
                                 + 0.0292236952228786 w_k1_FDX_s014
                                 + 0.0108658928942706 w_k1_FDX_s015
                                 + 0.0245350347078915 w_k1_FDX_s016
                                 + 0.0367621028056454 w_k1_FDX_s017
                                 + 0.0164034141809854 w_k1_FDX_s100
                                 + 0.0486281536327898 w_k1_FDX_s101
                                 + 0.0286262173136031 w_k1_FDX_s102
                                 + 0.0517106160079654 w_k1_FDX_s103
                                 + 0.0228664469748556 w_k1_FDX_s104
                                 + 0.0226441940714327 w_k1_FDX_s105
                                 + 0.0379504895030056 w_k1_FDX_s106
                                 + 0.0123765099416229 w_k1_FDX_s107
                                 + 0.0224731960815443 w_k1_FDX_s110
                                 + 0.0243982044205975 w_k1_FDX_s111
                                 + 0.0240407988963931 w_k1_FDX_s112
                                 + 0.102443958437897 w_k1_FDX_s113
                                 + 0.0858897465730507 w_k1_FDX_s114
                                 + 0.0236939170734738 w_k1_FDX_s115
                                 + 0.0225547568746076 w_k1_FDX_s116
                                 + 0.0232093372321306 w_k1_FDX_s117
                                 + 0.0179498137515188 w_k1_IBM_s000
                                 + 0.0165371911186641 w_k1_IBM_s001
                                 + 0.0339250832860523 w_k1_IBM_s002
                                 + 0.0185363846305663 w_k1_IBM_s003
                                 + 0.0650210344197848 w_k1_IBM_s004
                                 + 0.0259471492108613 w_k1_IBM_s005
                                 + 0.0182096208309802 w_k1_IBM_s006
                                 + 0.0147754201119447 w_k1_IBM_s007
                                 + 0.0372486061473579 w_k1_IBM_s010
                                 + 0.0512458246114862 w_k1_IBM_s011
                                 + 0.0297308203490482 w_k1_IBM_s012
                                 + 0.022733303991602 w_k1_IBM_s013
                                 + 0.0295384465394842 w_k1_IBM_s014
                                 + 0.0112710093293741 w_k1_IBM_s015
                                 + 0.0260537067250185 w_k1_IBM_s016
                                 + 0.0373843330349595 w_k1_IBM_s017
                                 + 0.0168127204241278 w_k1_IBM_s100
                                 + 0.0462417320702065 w_k1_IBM_s101
                                 + 0.0299963836980184 w_k1_IBM_s102
                                 + 0.0514165542613469 w_k1_IBM_s103
                                 + 0.02282782811976 w_k1_IBM_s104
                                 + 0.0235209644463595 w_k1_IBM_s105
                                 + 0.0386850207960315 w_k1_IBM_s106
                                 + 0.0124896234079111 w_k1_IBM_s107
                                 + 0.022591152734281 w_k1_IBM_s110
                                 + 0.0232435713509822 w_k1_IBM_s111
                                 + 0.0247442132640899 w_k1_IBM_s112
                                 + 0.104389161437202 w_k1_IBM_s113
                                 + 0.0861030845930649 w_k1_IBM_s114
                                 + 0.0257375711704657 w_k1_IBM_s115
                                 + 0.0239887142307008 w_k1_IBM_s116
                                 + 0.0236177622274369 w_k1_IBM_s117
                                 + 0.0174998305575345 w_k1_INTC_s000
                                 + 0.0174492900879235 w_k1_INTC_s001
                                 + 0.0325736523466242 w_k1_INTC_s002
                                 + 0.0174607028414036 w_k1_INTC_s003
                                 + 0.0596748963327893 w_k1_INTC_s004
                                 + 0.0232520206536905 w_k1_INTC_s005
                                 + 0.0183920283727633 w_k1_INTC_s006
                                 + 0.0119127416761886 w_k1_INTC_s007
                                 + 0.0374830210191977 w_k1_INTC_s010
                                 + 0.0526215052062228 w_k1_INTC_s011
                                 + 0.0287701627348628 w_k1_INTC_s012
                                 + 0.022697562695652 w_k1_INTC_s013
                                 + 0.0373156927347197 w_k1_INTC_s014
                                 + 0.00955148264233397 w_k1_INTC_s015
                                 + 0.0247349062404936 w_k1_INTC_s016
                                 + 0.0337459825891367 w_k1_INTC_s017
                                 + 0.0159298030570811 w_k1_INTC_s100
                                 + 0.0532752352964714 w_k1_INTC_s101
                                 + 0.0283316839389218 w_k1_INTC_s102
                                 + 0.0466288927599757 w_k1_INTC_s103
                                 + 0.0244832305004155 w_k1_INTC_s104
                                 + 0.0239988395665809 w_k1_INTC_s105
                                 + 0.0386879063196354 w_k1_INTC_s106
                                 + 0.0122996169160982 w_k1_INTC_s107
                                 + 0.0212835430574493 w_k1_INTC_s110
                                 + 0.0266172506183867 w_k1_INTC_s111
                                 + 0.0235773361355397 w_k1_INTC_s112
                                 + 0.105865164141382 w_k1_INTC_s113
                                 + 0.0888352088577995 w_k1_INTC_s114
                                 + 0.0276727049679318 w_k1_INTC_s115
                                 + 0.0257020521169553 w_k1_INTC_s116
                                 + 0.0262257427553148 w_k1_INTC_s117
                                 + 0.0185123565586435 w_k1_KO_s000
                                 + 0.0154683337515857 w_k1_KO_s001
                                 + 0.0356919924231737 w_k1_KO_s002
                                 + 0.0180198251554109 w_k1_KO_s003
                                 + 0.0660639970777692 w_k1_KO_s004
                                 + 0.0247567878747961 w_k1_KO_s005
                                 + 0.0189833788383908 w_k1_KO_s006
                                 + 0.01366252888442 w_k1_KO_s007
                                 + 0.0345997033585557 w_k1_KO_s010
                                 + 0.0521991019525357 w_k1_KO_s011
                                 + 0.0335338220100019 w_k1_KO_s012
                                 + 0.0235277268829775 w_k1_KO_s013
                                 + 0.0350438137618111 w_k1_KO_s014
                                 + 0.0104402934084917 w_k1_KO_s015
                                 + 0.0256048287824038 w_k1_KO_s016
                                 + 0.0355482132739564 w_k1_KO_s017
                                 + 0.0164616891228655 w_k1_KO_s100
                                 + 0.0453276844262143 w_k1_KO_s101
                                 + 0.0304572564049921 w_k1_KO_s102
                                 + 0.0558339644176059 w_k1_KO_s103
                                 + 0.0236674696183628 w_k1_KO_s104
                                 + 0.0240175983810061 w_k1_KO_s105
                                 + 0.0357025672592956 w_k1_KO_s106
                                 + 0.0130033779406876 w_k1_KO_s107
                                 + 0.0206432434357258 w_k1_KO_s110
                                 + 0.0218644395996139 w_k1_KO_s111
                                 + 0.0243895624498385 w_k1_KO_s112
                                 + 0.105792819095285 w_k1_KO_s113
                                 + 0.0788782864662745 w_k1_KO_s114
                                 + 0.0261488527734131 w_k1_KO_s115
                                 + 0.0243059404330148 w_k1_KO_s116
                                 + 0.0221071994378507 w_k1_KO_s117
                                 + 0.018143950602064 w_k1_MAT_s000
                                 + 0.0155706652077308 w_k1_MAT_s001
                                 + 0.0330923425438131 w_k1_MAT_s002
                                 + 0.0164015840684662 w_k1_MAT_s003
                                 + 0.0795578659867042 w_k1_MAT_s004
                                 + 0.0224434205305969 w_k1_MAT_s005
                                 + 0.0178808360422621 w_k1_MAT_s006
                                 + 0.0155244960797045 w_k1_MAT_s007
                                 + 0.0420252706252405 w_k1_MAT_s010
                                 + 0.0515194064552589 w_k1_MAT_s011
                                 + 0.0264556098543407 w_k1_MAT_s012
                                 + 0.0223865009053017 w_k1_MAT_s013
                                 + 0.0304997967884872 w_k1_MAT_s014
                                 + 0.0111198455034656 w_k1_MAT_s015
                                 + 0.0222835662218884 w_k1_MAT_s016
                                 + 0.0393610792383335 w_k1_MAT_s017
                                 + 0.0176260562184978 w_k1_MAT_s100
                                 + 0.049561170015577 w_k1_MAT_s101
                                 + 0.0316563694460063 w_k1_MAT_s102
                                 + 0.0494214370011217 w_k1_MAT_s103
                                 + 0.0200845394031895 w_k1_MAT_s104
                                 + 0.0276653684931229 w_k1_MAT_s105
                                 + 0.0326634677093333 w_k1_MAT_s106
                                 + 0.0151621312135527 w_k1_MAT_s107
                                 + 0.0211741849484018 w_k1_MAT_s110
                                 + 0.026633489500416 w_k1_MAT_s111
                                 + 0.0256965341455095 w_k1_MAT_s112
                                 + 0.0907883554787618 w_k1_MAT_s113
                                 + 0.0855678266930418 w_k1_MAT_s114
                                 + 0.0236216324367357 w_k1_MAT_s115
                                 + 0.0241497645941124 w_k1_MAT_s116
                                 + 0.0204403090991308 w_k1_MAT_s117
                                 + 0.0179803607612764 w_k1_MSFT_s000
                                 + 0.0165828637075858 w_k1_MSFT_s001
                                 + 0.0313697480940984 w_k1_MSFT_s002
                                 + 0.0174046843636612 w_k1_MSFT_s003
                                 + 0.0594229105204227 w_k1_MSFT_s004
                                 + 0.0249639829094555 w_k1_MSFT_s005
                                 + 0.0181192318236279 w_k1_MSFT_s006
                                 + 0.0132487066731543 w_k1_MSFT_s007
                                 + 0.0397659786797768 w_k1_MSFT_s010
                                 + 0.0492293577612911 w_k1_MSFT_s011
                                 + 0.0268812395618091 w_k1_MSFT_s012
                                 + 0.0239229996752021 w_k1_MSFT_s013
                                 + 0.0358881049117719 w_k1_MSFT_s014
                                 + 0.0106816431552248 w_k1_MSFT_s015
                                 + 0.0235384528304587 w_k1_MSFT_s016
                                 + 0.0338140793606754 w_k1_MSFT_s017
                                 + 0.0160024254466655 w_k1_MSFT_s100
                                 + 0.047503989604263 w_k1_MSFT_s101
                                 + 0.0266637387195374 w_k1_MSFT_s102
                                 + 0.0557706873890347 w_k1_MSFT_s103
                                 + 0.0232411281196962 w_k1_MSFT_s104
                                 + 0.0249296558012383 w_k1_MSFT_s105
                                 + 0.0350643322496802 w_k1_MSFT_s106
                                 + 0.0121246702615009 w_k1_MSFT_s107
                                 + 0.0206432618021287 w_k1_MSFT_s110
                                 + 0.0227063773472309 w_k1_MSFT_s111
                                 + 0.0242024729602664 w_k1_MSFT_s112
                                 + 0.0978156842511004 w_k1_MSFT_s113
                                 + 0.0847149722153448 w_k1_MSFT_s114
                                 + 0.0245374967942318 w_k1_MSFT_s115
                                 + 0.022354499842104 w_k1_MSFT_s116
                                 + 0.0221742324142477 w_k1_MSFT_s117
                                 + 0.0191507686499325 w_k1_PG_s000
                                 + 0.0154594323740053 w_k1_PG_s001
                                 + 0.0348658418154286 w_k1_PG_s002
                                 + 0.018905063283747 w_k1_PG_s003
                                 + 0.0630406629476857 w_k1_PG_s004
                                 + 0.026158921239281 w_k1_PG_s005
                                 + 0.0180369532518003 w_k1_PG_s006
                                 + 0.0130500198699432 w_k1_PG_s007
                                 + 0.0372399676759436 w_k1_PG_s010
                                 + 0.0538404441046233 w_k1_PG_s011
                                 + 0.031886309604938 w_k1_PG_s012
                                 + 0.0234414976833024 w_k1_PG_s013
                                 + 0.028741137086626 w_k1_PG_s014
                                 + 0.0109401656362651 w_k1_PG_s015
                                 + 0.0250621025794854 w_k1_PG_s016
                                 + 0.0325935823098447 w_k1_PG_s017
                                 + 0.0159679693816409 w_k1_PG_s100
                                 + 0.0470793369243313 w_k1_PG_s101
                                 + 0.0279558819213098 w_k1_PG_s102
                                 + 0.0523571198213519 w_k1_PG_s103
                                 + 0.0219555236242369 w_k1_PG_s104
                                 + 0.0231105967852028 w_k1_PG_s105
                                 + 0.0393965927936019 w_k1_PG_s106
                                 + 0.0125315539426922 w_k1_PG_s107
                                 + 0.0229576378249492 w_k1_PG_s110
                                 + 0.023485256074732 w_k1_PG_s111
                                 + 0.0232932686407523 w_k1_PG_s112
                                 + 0.097169915334539 w_k1_PG_s113
                                 + 0.0823543531361623 w_k1_PG_s114
                                 + 0.0231140669883517 w_k1_PG_s115
                                 + 0.0203486453365478 w_k1_PG_s116
                                 + 0.0229700124478868 w_k1_PG_s117 >= 1.16
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
