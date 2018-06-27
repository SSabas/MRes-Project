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
 w_k1_ADM_s0_return_constraint:  1.08948 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.94569 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.21286 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.960724 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.805753 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.07824 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.095 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.16869 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.09796 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.15834 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.757061 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.05963 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.784667 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.28767 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.02128 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.960512 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.871343 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.17939 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.1753 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.00721 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.983332 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.731744 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.851541 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.857513 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.841588 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.14573 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.49446 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.934855 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.23422 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.0689 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.06592 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.811163 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.20009 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.691056 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.01093 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.870947 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.01175 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.2061 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.17733 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.809915 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.32528 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.825557 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.895759 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23915 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.21218 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.17301 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.38159 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.1087 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.731236 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.959327 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.864968 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.69127 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.31631 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.936806 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.806016 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.67605 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.40827 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.12838 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.09682 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.647848 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.723279 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.671498 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.842572 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.05868 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.28173 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.722439 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.27727 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.875952 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.05182 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.803987 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.39181 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.698666 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.20519 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.750964 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.07179 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.38352 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.20801 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.806505 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.41074 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.848297 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.871988 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.20371 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.08256 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.45644 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.34283 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.35043 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.620605 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.918553 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.783772 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.63712 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.24329 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.933178 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.869151 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 2.04853 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.2709 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.86499 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.19977 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.561667 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.674628 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.587091 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.853996 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.23341 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.51287 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.568208 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.967524 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.873186 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.15016 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.781857 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.27551 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.546899 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.900544 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.783777 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.34548 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.46067 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.12027 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.87009 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.28025 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.886996 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.930523 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.11558 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.08953 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.29329 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.02727 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.14466 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.779647 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.01495 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.825778 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.41753 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.01062 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.887603 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.981575 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.55632 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.27364 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.975414 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.20627 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.737256 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.682679 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.779408 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.835513 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.13481 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.45541 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.678332 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.06822 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.949091 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.08265 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.738892 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.39539 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.664306 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.02958 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.858448 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.23848 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.14365 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13999 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.827674 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.25149 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.974422 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.806878 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.28348 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.834869 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.839841 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.8514 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.18486 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.59394 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.657118 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.1147 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.33788 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.65385 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.1271 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.57384 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.14777 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.09197 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.80126 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.660218 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.7096 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.44799 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.550655 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.20561 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.51582 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.889894 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.911467 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.06306 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.823518 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.01254 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.31662 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.685587 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.84742 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.891002 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.805426 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.817942 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.40635 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02018 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.874463 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.3623 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.882719 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.926779 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.11203 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.937793 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.10088 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.834814 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06059 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.701449 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.817035 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.804402 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.18663 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.17444 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.04458 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.986761 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.52493 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.36121 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.03218 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.22253 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.80092 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.787277 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.813643 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.966133 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.19171 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.3467 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.759787 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.05931 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.860529 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.06073 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.815651 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.16265 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.704918 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.07039 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.930581 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.37119 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.28864 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18195 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.887518 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.19108 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.96677 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.803833 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.06653 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.949479 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.10661 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.41434 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.10346 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.607514 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.798137 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.887799 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.37934 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.24513 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.959613 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.676379 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.25243 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.0851 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.772954 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.889692 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.688672 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.0598 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.645592 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.00435 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.33409 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.19767 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.779161 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.01314 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.936657 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.10428 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.03367 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.91581 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.684174 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.907723 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.888597 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.24405 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.51567 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15778 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.815507 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.19099 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.02459 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.777691 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.23918 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.12761 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.0088 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.48574 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.46185 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.623651 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.709426 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.05814 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.49773 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.68479 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.878256 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.615038 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.29894 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.26002 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.866685 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.642735 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.696509 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.4845 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.683459 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.987624 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.7353 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.944315 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.910246 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.25148 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.867898 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.924998 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.42872 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.901804 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.766907 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04185 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.852133 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.83052 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.45854 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.07196 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.90398 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.25325 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.91463 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.907972 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.963996 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.09372 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.3109 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.969077 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.19512 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.788539 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.929965 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.87769 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.21392 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.977494 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.971616 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.00346 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.52567 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.234 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.897515 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.19437 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.759104 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.796612 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.782181 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.848675 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.13126 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.34205 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.714047 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.06861 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.963915 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.20178 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.916612 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.32109 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.77882 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.13491 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.987177 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.39248 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.36542 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06013 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.913408 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.20599 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.918411 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.871518 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10813 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.01733 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.10634 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.11774 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06036 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.811217 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.924751 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.897261 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.26497 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.1991 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.08234 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.954074 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.46918 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.19245 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.03688 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.0839 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.837669 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.96475 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.846659 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.946647 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.20084 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.30811 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.885754 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04381 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.01924 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.983021 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.932498 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.14357 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.77843 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00822 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.884901 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.13668 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.09879 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0919999408629148 z_k1_s0000
                                 - 0.0949999332899732 z_k1_s0001
                                 - 0.13999981969585 z_k1_s0002
                                 - 0.325000045096942 z_k1_s0003
                                 - 0.153999784355457 z_k1_s0004
                                 - 0.255999007576081 z_k1_s0005
                                 - 0.25299918824881 z_k1_s0006
                                 - 0.416000334683747 z_k1_s0007
                                 - 0.158999841491905 z_k1_s0010
                                 - 0.192000076744709 z_k1_s0011
                                 - 0.0929998877858485 z_k1_s0012
                                 - 0.170999927038379 z_k1_s0013
                                 - 0.108999872647926 z_k1_s0014
                                 - 0.279999412086643 z_k1_s0015
                                 - 0.1960001052602 z_k1_s0016
                                 - 0.0919998806569757 z_k1_s0017
                                 - 0.336999081994505 z_k1_s0020
                                 - 0.223999555390825 z_k1_s0021
                                 - 0.161999623095211 z_k1_s0022
                                 - 0.191999840334751 z_k1_s0023
                                 - 0.161999623095211 z_k1_s0024
                                 - 0.0709999974674765 z_k1_s0025
                                 - 0.230999347747 z_k1_s0026
                                 - 0.171000463266227 z_k1_s0027
                                 - 0.244998665073997 z_k1_s0030
                                 - 0.390997458197568 z_k1_s0031
                                 - 0.163999307630224 z_k1_s0032
                                 - 0.249998503688379 z_k1_s0033
                                 - 0.277999352720723 z_k1_s0034
                                 - 0.53999790728157 z_k1_s0035
                                 - 0.150999289034879 z_k1_s0036
                                 - 0.171999268512211 z_k1_s0037
                                 - 0.173999780469066 z_k1_s0040
                                 - 0.308000501610426 z_k1_s0041
                                 - 0.137999658975744 z_k1_s0042
                                 - 0.316999206610966 z_k1_s0043
                                 - 0.424999801590546 z_k1_s0044
                                 - 0.15999969480506 z_k1_s0045
                                 - 0.372999088267756 z_k1_s0046
                                 - 0.40999888659477 z_k1_s0047
                                 - 0.266999936257694 z_k1_s0050
                                 - 0.408998297646639 z_k1_s0051
                                 - 0.400001462618792 z_k1_s0052
                                 - 0.651998379427284 z_k1_s0053
                                 - 0.361000143893064 z_k1_s0054
                                 - 0.288000197725243 z_k1_s0055
                                 - 0.902999967900801 z_k1_s0056
                                 - 0.608001756620407 z_k1_s0057
                                 - 0.465999559754135 z_k1_s0060
                                 - 0.536001559221458 z_k1_s0061
                                 - 0.574002289017641 z_k1_s0062
                                 - 0.416999819728704 z_k1_s0063
                                 - 1.05500049908439 z_k1_s0064
                                 - 0.530002099096413 z_k1_s0065
                                 - 0.217000389742769 z_k1_s0066
                                 - 0.354000509610655 z_k1_s0067
                                 - 0.282000285232296 z_k1_s0070
                                 - 0.217000256224801 z_k1_s0071
                                 - 0.108999775412952 z_k1_s0072
                                 - 0.138999777416427 z_k1_s0073
                                 - 0.13799974281636 z_k1_s0074
                                 - 0.147999792817449 z_k1_s0075
                                 - 0.325000145037478 z_k1_s0076
                                 - 0.121999781214451 z_k1_s0077
                                 - 0.790005107384633 z_k1_s0100
                                 - 1.55700274513308 z_k1_s0101
                                 - 0.71400145558388 z_k1_s0102
                                 - 1.0150042978112 z_k1_s0103
                                 - 0.655001866619199 z_k1_s0104
                                 - 1.23100024129234 z_k1_s0105
                                 - 0.919001960142357 z_k1_s0106
                                 - 0.572001320586359 z_k1_s0107
                                 - 0.481001285785705 z_k1_s0110
                                 - 0.772001714270152 z_k1_s0111
                                 - 0.609001763027354 z_k1_s0112
                                 - 1.76900635874922 z_k1_s0113
                                 - 1.09300478583477 z_k1_s0114
                                 - 0.716002209347445 z_k1_s0115
                                 - 1.29299930497799 z_k1_s0116
                                 - 1.93000861718623 z_k1_s0117
                                 - 0.170000013605885 z_k1_s0120
                                 - 0.22999994737857 z_k1_s0121
                                 - 0.335998856327252 z_k1_s0122
                                 - 0.34599936853922 z_k1_s0123
                                 - 0.28199947138168 z_k1_s0124
                                 - 0.393000085435943 z_k1_s0125
                                 - 0.4739991628543 z_k1_s0126
                                 - 0.184000006202843 z_k1_s0127
                                 - 0.147000034956778 z_k1_s0130
                                 - 0.279000462160991 z_k1_s0131
                                 - 0.136999982621602 z_k1_s0132
                                 - 0.297000066784101 z_k1_s0133
                                 - 0.241999496490512 z_k1_s0134
                                 - 0.234000509102818 z_k1_s0135
                                 - 0.315999544830672 z_k1_s0136
                                 - 0.231000888832433 z_k1_s0137
                                 - 1.20500375793352 z_k1_s0140
                                 - 0.37800101590186 z_k1_s0141
                                 - 0.530999681130346 z_k1_s0142
                                 - 0.270000928401979 z_k1_s0143
                                 - 0.287000885227703 z_k1_s0144
                                 - 0.878000720102765 z_k1_s0145
                                 - 0.299001105217254 z_k1_s0146
                                 - 0.883001403141975 z_k1_s0147
                                 - 1.50699952235562 z_k1_s0150
                                 - 0.455000463925913 z_k1_s0151
                                 - 0.327000303813971 z_k1_s0152
                                 - 1.52900201780919 z_k1_s0153
                                 - 0.409000310729315 z_k1_s0154
                                 - 0.733000218599548 z_k1_s0155
                                 - 0.520000880030052 z_k1_s0156
                                 - 0.642002697057574 z_k1_s0157
                                 - 1.23499826911151 z_k1_s0160
                                 - 0.47099977712135 z_k1_s0161
                                 - 0.375999459467165 z_k1_s0162
                                 - 0.87000044150974 z_k1_s0163
                                 - 0.257999611289725 z_k1_s0164
                                 - 0.428001237238038 z_k1_s0165
                                 - 0.252999858949711 z_k1_s0166
                                 - 0.294999620443599 z_k1_s0167
                                 - 0.999997154652465 z_k1_s0170
                                 - 0.33599986367502 z_k1_s0171
                                 - 1.28400167470042 z_k1_s0172
                                 - 0.341000105608218 z_k1_s0173
                                 - 0.950001139318868 z_k1_s0174
                                 - 1.43299710095268 z_k1_s0175
                                 - 0.735998725536006 z_k1_s0176
                                 - 0.324000051515152 z_k1_s0177
                                 - 0.0620001079391899 z_k1_s1000
                                 - 0.131000072787481 z_k1_s1001
                                 - 0.174999704730958 z_k1_s1002
                                 - 0.118999974257234 z_k1_s1003
                                 - 0.222000181807853 z_k1_s1004
                                 - 0.12199972538951 z_k1_s1005
                                 - 0.116000223124958 z_k1_s1006
                                 - 0.146999839494096 z_k1_s1007
                                 - 0.105000087949499 z_k1_s1010
                                 - 0.169000279964614 z_k1_s1011
                                 - 0.124000108328937 z_k1_s1012
                                 - 0.16000048003779 z_k1_s1013
                                 - 0.0710000884916041 z_k1_s1014
                                 - 0.2270002928281 z_k1_s1015
                                 - 0.105000087949499 z_k1_s1016
                                 - 0.211000127624165 z_k1_s1017
                                 - 0.451000953009102 z_k1_s1020
                                 - 0.453000557265097 z_k1_s1021
                                 - 0.184000628367434 z_k1_s1022
                                 - 0.103000344448595 z_k1_s1023
                                 - 0.121000428863959 z_k1_s1024
                                 - 0.17600048423805 z_k1_s1025
                                 - 0.2010005268536 z_k1_s1026
                                 - 0.230000545583542 z_k1_s1027
                                 - 0.225000744577216 z_k1_s1030
                                 - 0.316002082714081 z_k1_s1031
                                 - 0.25900059851881 z_k1_s1032
                                 - 0.386002409125042 z_k1_s1033
                                 - 0.538001290297895 z_k1_s1034
                                 - 0.219000716599134 z_k1_s1035
                                 - 0.488001057147209 z_k1_s1036
                                 - 0.61500134474908 z_k1_s1037
                                 - 0.254999083643948 z_k1_s1040
                                 - 0.178999606659374 z_k1_s1041
                                 - 0.189999672859492 z_k1_s1042
                                 - 0.441000203027756 z_k1_s1043
                                 - 0.180999618695759 z_k1_s1044
                                 - 0.231999435424549 z_k1_s1045
                                 - 0.793997425459428 z_k1_s1046
                                 - 0.178999606659374 z_k1_s1047
                                 - 0.290000084328835 z_k1_s1050
                                 - 0.248000047894988 z_k1_s1051
                                 - 0.195999584690893 z_k1_s1052
                                 - 0.184000828567786 z_k1_s1053
                                 - 0.170000055489321 z_k1_s1054
                                 - 0.403000516829652 z_k1_s1055
                                 - 0.165000193304436 z_k1_s1056
                                 - 0.100000053299615 z_k1_s1057
                                 - 0.438001191179053 z_k1_s1060
                                 - 0.102000219015575 z_k1_s1061
                                 - 0.237000383538925 z_k1_s1062
                                 - 0.363999444516145 z_k1_s1063
                                 - 0.369000917671254 z_k1_s1064
                                 - 0.364999739147167 z_k1_s1065
                                 - 0.315000049620606 z_k1_s1066
                                 - 0.317000638882649 z_k1_s1067
                                 - 0.564997707281935 z_k1_s1070
                                 - 0.690997288427942 z_k1_s1071
                                 - 0.385999852638718 z_k1_s1072
                                 - 0.744997855061526 z_k1_s1073
                                 - 0.78099997381627 z_k1_s1074
                                 - 1.05499839973733 z_k1_s1075
                                 - 0.776999158066069 z_k1_s1076
                                 - 0.223000037999242 z_k1_s1077
                                 - 0.225000053226459 z_k1_s1100
                                 - 0.31999924814469 z_k1_s1101
                                 - 0.244000284209919 z_k1_s1102
                                 - 0.139999989563151 z_k1_s1103
                                 - 0.249000241837195 z_k1_s1104
                                 - 0.207999021294281 z_k1_s1105
                                 - 0.176999872004895 z_k1_s1106
                                 - 0.396999771604172 z_k1_s1107
                                 - 0.304998654703965 z_k1_s1110
                                 - 0.268999618285679 z_k1_s1111
                                 - 0.433000070453246 z_k1_s1112
                                 - 0.694998788225298 z_k1_s1113
                                 - 0.210999761079787 z_k1_s1114
                                 - 0.295999309298999 z_k1_s1115
                                 - 0.35000089772353 z_k1_s1116
                                 - 0.19899971454071 z_k1_s1117
                                 - 0.293998755583285 z_k1_s1120
                                 - 0.126999557658495 z_k1_s1121
                                 - 0.22000031935808 z_k1_s1122
                                 - 0.427999367900198 z_k1_s1123
                                 - 0.370999076258018 z_k1_s1124
                                 - 0.120999757848134 z_k1_s1125
                                 - 0.213000100297281 z_k1_s1126
                                 - 0.31499941276568 z_k1_s1127
                                 - 0.338998546215953 z_k1_s1130
                                 - 0.470998610138676 z_k1_s1131
                                 - 0.151999404855639 z_k1_s1132
                                 - 0.364999421713045 z_k1_s1133
                                 - 0.302997881602792 z_k1_s1134
                                 - 0.302997881602792 z_k1_s1135
                                 - 0.17899931006746 z_k1_s1136
                                 - 0.26099829271687 z_k1_s1137
                                 - 0.137999945805878 z_k1_s1140
                                 - 0.140999969569034 z_k1_s1141
                                 - 0.163999878586731 z_k1_s1142
                                 - 0.395000397150545 z_k1_s1143
                                 - 0.35199961947891 z_k1_s1144
                                 - 0.171000043300694 z_k1_s1145
                                 - 0.0869998696320256 z_k1_s1146
                                 - 0.190999109055737 z_k1_s1147
                                 - 0.641997325994232 z_k1_s1150
                                 - 0.475996942594764 z_k1_s1151
                                 - 0.216999333930182 z_k1_s1152
                                 - 0.840997050825262 z_k1_s1153
                                 - 0.506999386503853 z_k1_s1154
                                 - 0.295998760611072 z_k1_s1155
                                 - 0.192999274300154 z_k1_s1156
                                 - 0.312998875452903 z_k1_s1157
                                 - 0.285999555588401 z_k1_s1160
                                 - 0.105999901279533 z_k1_s1161
                                 - 0.0889999208214903 z_k1_s1162
                                 - 0.285999555588401 z_k1_s1163
                                 - 0.387999438336659 z_k1_s1164
                                 - 0.267000155663973 z_k1_s1165
                                 - 0.208999297161233 z_k1_s1166
                                 - 0.335000077496145 z_k1_s1167
                                 - 0.506999813102548 z_k1_s1170
                                 - 0.468001094864744 z_k1_s1171
                                 - 0.244000250460035 z_k1_s1172
                                 - 0.255000047360548 z_k1_s1173
                                 - 0.500001627394265 z_k1_s1174
                                 - 0.950999478819458 z_k1_s1175
                                 - 0.399000384242008 z_k1_s1176
                                 - 0.795000486865468 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.928455 w_k1_ADM_s000
                                 + 0.858432 w_k1_AXP_s000
                                 + 0.828813 w_k1_DIS_s000
                                 + 0.990459 w_k1_FDX_s000
                                 + 0.666072 w_k1_IBM_s000
                                 + 1.00683 w_k1_INTC_s000
                                 + 0.89079 w_k1_KO_s000
                                 + 0.762152 w_k1_MAT_s000
                                 + 0.993529 w_k1_MSFT_s000
                                 + 0.969304 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.23473 w_k1_ADM_s000
                                 + 1.15229 w_k1_AXP_s000
                                 + 1.24089 w_k1_DIS_s000
                                 + 1.30309 w_k1_FDX_s000
                                 + 0.630518 w_k1_IBM_s000
                                 + 1.30161 w_k1_INTC_s000
                                 + 1.02224 w_k1_KO_s000
                                 + 0.548213 w_k1_MAT_s000
                                 + 1.00528 w_k1_MSFT_s000
                                 + 1.20379 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.793841 w_k1_ADM_s000
                                 + 0.7073 w_k1_AXP_s000 + 0.67589 w_k1_DIS_s000
                                 + 0.771071 w_k1_FDX_s000
                                 + 0.610473 w_k1_IBM_s000
                                 + 0.882964 w_k1_INTC_s000
                                 + 0.778667 w_k1_KO_s000
                                 + 0.686979 w_k1_MAT_s000
                                 + 0.862456 w_k1_MSFT_s000
                                 + 0.871715 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.17304 w_k1_ADM_s000
                                 + 0.849129 w_k1_AXP_s000
                                 + 1.18615 w_k1_DIS_s000
                                 + 1.13004 w_k1_FDX_s000
                                 + 0.995899 w_k1_IBM_s000
                                 + 1.21763 w_k1_INTC_s000
                                 + 1.17428 w_k1_KO_s000
                                 + 0.800824 w_k1_MAT_s000
                                 + 1.13333 w_k1_MSFT_s000
                                 + 1.09307 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.94491 w_k1_ADM_s000
                                 + 0.788688 w_k1_AXP_s000
                                 + 1.10149 w_k1_DIS_s000
                                 + 0.994705 w_k1_FDX_s000
                                 + 0.813213 w_k1_IBM_s000
                                 + 0.955844 w_k1_INTC_s000
                                 + 1.01766 w_k1_KO_s000
                                 + 0.749523 w_k1_MAT_s000
                                 + 0.948402 w_k1_MSFT_s000
                                 + 0.897414 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.9186 w_k1_ADM_s000
                                 + 0.636498 w_k1_AXP_s000
                                 + 0.986323 w_k1_DIS_s000
                                 + 0.837206 w_k1_FDX_s000
                                 + 1.20039 w_k1_IBM_s000
                                 + 0.923583 w_k1_INTC_s000
                                 + 1.07824 w_k1_KO_s000 + 1.00609 w_k1_MAT_s000
                                 + 0.929614 w_k1_MSFT_s000
                                 + 1.0052 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.905573 w_k1_ADM_s000
                                 + 0.921759 w_k1_AXP_s000
                                 + 0.807707 w_k1_DIS_s000
                                 + 0.964595 w_k1_FDX_s000
                                 + 0.980073 w_k1_IBM_s000
                                 + 0.926211 w_k1_INTC_s000
                                 + 1.00472 w_k1_KO_s000 + 1.13144 w_k1_MAT_s000
                                 + 0.833301 w_k1_MSFT_s000
                                 + 0.972289 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.26935 w_k1_ADM_s000
                                 + 1.38615 w_k1_AXP_s000
                                 + 1.31336 w_k1_DIS_s000
                                 + 1.21287 w_k1_FDX_s000 + 1.3087 w_k1_IBM_s000
                                 + 1.36937 w_k1_INTC_s000
                                 + 1.32709 w_k1_KO_s000 + 1.4048 w_k1_MAT_s000
                                 + 1.15346 w_k1_MSFT_s000
                                 + 1.25537 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.755886 w_k1_ADM_s001
                                 + 0.825795 w_k1_AXP_s001
                                 + 0.843176 w_k1_DIS_s001
                                 + 0.782199 w_k1_FDX_s001
                                 + 1.15766 w_k1_IBM_s001
                                 + 0.943068 w_k1_INTC_s001
                                 + 1.16178 w_k1_KO_s001 + 1.08188 w_k1_MAT_s001
                                 + 0.810074 w_k1_MSFT_s001
                                 + 0.862608 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.34422 w_k1_ADM_s001
                                 + 1.09232 w_k1_AXP_s001
                                 + 1.28549 w_k1_DIS_s001
                                 + 1.22693 w_k1_FDX_s001
                                 + 0.862592 w_k1_IBM_s001
                                 + 1.20619 w_k1_INTC_s001
                                 + 0.989748 w_k1_KO_s001
                                 + 0.856884 w_k1_MAT_s001
                                 + 1.17312 w_k1_MSFT_s001
                                 + 0.974082 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.23598 w_k1_ADM_s001
                                 + 0.85947 w_k1_AXP_s001
                                 + 1.20996 w_k1_DIS_s001 + 1.0509 w_k1_FDX_s001
                                 + 0.776452 w_k1_IBM_s001
                                 + 1.25174 w_k1_INTC_s001
                                 + 0.94911 w_k1_KO_s001
                                 + 0.607752 w_k1_MAT_s001
                                 + 1.21077 w_k1_MSFT_s001
                                 + 0.908308 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.896743 w_k1_ADM_s001
                                 + 0.934206 w_k1_AXP_s001
                                 + 0.649856 w_k1_DIS_s001
                                 + 0.909916 w_k1_FDX_s001
                                 + 0.692742 w_k1_IBM_s001
                                 + 0.974707 w_k1_INTC_s001
                                 + 0.807835 w_k1_KO_s001
                                 + 1.00205 w_k1_MAT_s001
                                 + 0.993816 w_k1_MSFT_s001
                                 + 0.89964 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.43023 w_k1_ADM_s001
                                 + 1.74826 w_k1_AXP_s001
                                 + 1.48003 w_k1_DIS_s001
                                 + 1.54845 w_k1_FDX_s001
                                 + 1.25181 w_k1_IBM_s001
                                 + 1.34322 w_k1_INTC_s001
                                 + 1.39046 w_k1_KO_s001 + 1.78452 w_k1_MAT_s001
                                 + 1.18231 w_k1_MSFT_s001
                                 + 1.24264 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.9019 w_k1_ADM_s001
                                 + 1.03511 w_k1_AXP_s001
                                 + 1.03499 w_k1_DIS_s001
                                 + 1.00587 w_k1_FDX_s001
                                 + 1.01093 w_k1_IBM_s001
                                 + 0.959039 w_k1_INTC_s001
                                 + 0.967939 w_k1_KO_s001
                                 + 1.17265 w_k1_MAT_s001
                                 + 0.908738 w_k1_MSFT_s001
                                 + 1.10979 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.791557 w_k1_ADM_s001
                                 + 0.718595 w_k1_AXP_s001
                                 + 0.873538 w_k1_DIS_s001
                                 + 0.870799 w_k1_FDX_s001
                                 + 0.816448 w_k1_IBM_s001
                                 + 0.840498 w_k1_INTC_s001
                                 + 0.967051 w_k1_KO_s001
                                 + 0.811395 w_k1_MAT_s001
                                 + 0.987168 w_k1_MSFT_s001
                                 + 0.98761 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.14698 w_k1_ADM_s001
                                 + 1.15141 w_k1_AXP_s001
                                 + 1.31256 w_k1_DIS_s001
                                 + 0.944025 w_k1_FDX_s001
                                 + 1.29426 w_k1_IBM_s001
                                 + 1.00366 w_k1_INTC_s001
                                 + 1.21344 w_k1_KO_s001
                                 + 0.916887 w_k1_MAT_s001
                                 + 1.02477 w_k1_MSFT_s001
                                 + 1.13068 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.938304 w_k1_ADM_s002
                                 + 0.684496 w_k1_AXP_s002
                                 + 0.716269 w_k1_DIS_s002
                                 + 0.811741 w_k1_FDX_s002
                                 + 0.835608 w_k1_IBM_s002
                                 + 1.08803 w_k1_INTC_s002
                                 + 0.846997 w_k1_KO_s002
                                 + 0.579092 w_k1_MAT_s002
                                 + 0.891775 w_k1_MSFT_s002
                                 + 0.864768 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.30886 w_k1_ADM_s002
                                 + 1.10665 w_k1_AXP_s002
                                 + 1.05133 w_k1_DIS_s002
                                 + 1.15154 w_k1_FDX_s002
                                 + 0.80893 w_k1_IBM_s002
                                 + 1.17064 w_k1_INTC_s002
                                 + 1.08841 w_k1_KO_s002 + 0.73425 w_k1_MAT_s002
                                 + 1.20321 w_k1_MSFT_s002
                                 + 1.01686 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.771142 w_k1_ADM_s002
                                 + 0.710123 w_k1_AXP_s002
                                 + 0.616666 w_k1_DIS_s002
                                 + 0.755088 w_k1_FDX_s002
                                 + 1.07319 w_k1_IBM_s002
                                 + 0.932205 w_k1_INTC_s002
                                 + 0.877235 w_k1_KO_s002
                                 + 1.03553 w_k1_MAT_s002
                                 + 0.85681 w_k1_MSFT_s002
                                 + 0.847117 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.987303 w_k1_ADM_s002
                                 + 1.31215 w_k1_AXP_s002 + 1.1519 w_k1_DIS_s002
                                 + 1.09995 w_k1_FDX_s002
                                 + 1.28699 w_k1_IBM_s002
                                 + 1.15808 w_k1_INTC_s002 + 1.1072 w_k1_KO_s002
                                 + 1.10181 w_k1_MAT_s002
                                 + 1.13575 w_k1_MSFT_s002
                                 + 1.15932 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.1039 w_k1_ADM_s002
                                 + 1.49562 w_k1_AXP_s002
                                 + 1.68942 w_k1_DIS_s002
                                 + 1.17303 w_k1_FDX_s002 + 1.6811 w_k1_IBM_s002
                                 + 1.52365 w_k1_INTC_s002
                                 + 1.55015 w_k1_KO_s002 + 1.3676 w_k1_MAT_s002
                                 + 1.25079 w_k1_MSFT_s002
                                 + 1.09365 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.761487 w_k1_ADM_s002
                                 + 0.544329 w_k1_AXP_s002
                                 + 0.538827 w_k1_DIS_s002
                                 + 0.712966 w_k1_FDX_s002
                                 + 0.569902 w_k1_IBM_s002
                                 + 0.984432 w_k1_INTC_s002
                                 + 0.717029 w_k1_KO_s002
                                 + 0.41062 w_k1_MAT_s002
                                 + 0.975213 w_k1_MSFT_s002
                                 + 0.918469 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.907921 w_k1_ADM_s002
                                 + 1.02615 w_k1_AXP_s002
                                 + 1.43765 w_k1_DIS_s002
                                 + 1.35113 w_k1_FDX_s002
                                 + 0.990639 w_k1_IBM_s002
                                 + 1.53068 w_k1_INTC_s002
                                 + 1.18927 w_k1_KO_s002
                                 + 0.680411 w_k1_MAT_s002
                                 + 1.45509 w_k1_MSFT_s002
                                 + 1.17307 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.978464 w_k1_ADM_s002
                                 + 0.777544 w_k1_AXP_s002
                                 + 0.903253 w_k1_DIS_s002
                                 + 0.913198 w_k1_FDX_s002
                                 + 1.28966 w_k1_IBM_s002
                                 + 1.48183 w_k1_INTC_s002 + 1.2428 w_k1_KO_s002
                                 + 1.02276 w_k1_MAT_s002
                                 + 1.22488 w_k1_MSFT_s002
                                 + 1.05631 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.36242 w_k1_ADM_s003
                                 + 1.48375 w_k1_AXP_s003
                                 + 1.57412 w_k1_DIS_s003
                                 + 1.36507 w_k1_FDX_s003
                                 + 0.970172 w_k1_IBM_s003
                                 + 1.24109 w_k1_INTC_s003
                                 + 1.34231 w_k1_KO_s003 + 1.02937 w_k1_MAT_s003
                                 + 1.26393 w_k1_MSFT_s003
                                 + 1.28985 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.83949 w_k1_ADM_s003
                                 + 1.10544 w_k1_AXP_s003
                                 + 0.868364 w_k1_DIS_s003
                                 + 1.03221 w_k1_FDX_s003
                                 + 0.837445 w_k1_IBM_s003
                                 + 0.875043 w_k1_INTC_s003
                                 + 1.01978 w_k1_KO_s003
                                 + 0.901124 w_k1_MAT_s003
                                 + 0.826281 w_k1_MSFT_s003
                                 + 1.06876 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.40154 w_k1_ADM_s003
                                 + 1.07633 w_k1_AXP_s003
                                 + 1.10609 w_k1_DIS_s003
                                 + 1.15072 w_k1_FDX_s003
                                 + 0.727922 w_k1_IBM_s003
                                 + 1.1215 w_k1_INTC_s003 + 1.03912 w_k1_KO_s003
                                 + 0.526983 w_k1_MAT_s003
                                 + 1.09171 w_k1_MSFT_s003
                                 + 1.00816 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.06718 w_k1_ADM_s003
                                 + 1.09859 w_k1_AXP_s003
                                 + 1.20634 w_k1_DIS_s003
                                 + 1.04082 w_k1_FDX_s003
                                 + 0.993795 w_k1_IBM_s003
                                 + 1.12034 w_k1_INTC_s003
                                 + 1.31396 w_k1_KO_s003
                                 + 0.869244 w_k1_MAT_s003
                                 + 1.14938 w_k1_MSFT_s003
                                 + 1.11451 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.983724 w_k1_ADM_s003
                                 + 0.969441 w_k1_AXP_s003
                                 + 1.00581 w_k1_DIS_s003
                                 + 0.855835 w_k1_FDX_s003
                                 + 1.31689 w_k1_IBM_s003
                                 + 0.957083 w_k1_INTC_s003
                                 + 1.28453 w_k1_KO_s003 + 1.25103 w_k1_MAT_s003
                                 + 0.94255 w_k1_MSFT_s003
                                 + 1.05558 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.636411 w_k1_ADM_s003
                                 + 0.814344 w_k1_AXP_s003
                                 + 0.726166 w_k1_DIS_s003
                                 + 0.834808 w_k1_FDX_s003
                                 + 0.981646 w_k1_IBM_s003
                                 + 0.905157 w_k1_INTC_s003
                                 + 1.04128 w_k1_KO_s003
                                 + 0.967343 w_k1_MAT_s003
                                 + 0.82164 w_k1_MSFT_s003
                                 + 0.910918 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.921287 w_k1_ADM_s003
                                 + 0.664095 w_k1_AXP_s003
                                 + 0.602967 w_k1_DIS_s003
                                 + 0.714038 w_k1_FDX_s003
                                 + 0.659303 w_k1_IBM_s003
                                 + 0.765343 w_k1_INTC_s003
                                 + 0.885572 w_k1_KO_s003
                                 + 0.600043 w_k1_MAT_s003
                                 + 0.787303 w_k1_MSFT_s003
                                 + 0.839103 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.872846 w_k1_ADM_s003
                                 + 1.32277 w_k1_AXP_s003
                                 + 1.51627 w_k1_DIS_s003
                                 + 1.18232 w_k1_FDX_s003
                                 + 1.52809 w_k1_IBM_s003
                                 + 1.42491 w_k1_INTC_s003
                                 + 1.51751 w_k1_KO_s003 + 1.18017 w_k1_MAT_s003
                                 + 1.17294 w_k1_MSFT_s003
                                 + 1.17852 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.70251 w_k1_ADM_s004
                                 + 1.67683 w_k1_AXP_s004
                                 + 1.70165 w_k1_DIS_s004
                                 + 1.46737 w_k1_FDX_s004
                                 + 1.07526 w_k1_IBM_s004
                                 + 1.40363 w_k1_INTC_s004
                                 + 1.60494 w_k1_KO_s004 + 1.6537 w_k1_MAT_s004
                                 + 1.42443 w_k1_MSFT_s004
                                 + 1.28193 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.13359 w_k1_ADM_s004
                                 + 1.11361 w_k1_AXP_s004
                                 + 0.990611 w_k1_DIS_s004
                                 + 1.11364 w_k1_FDX_s004
                                 + 1.01229 w_k1_IBM_s004
                                 + 1.36192 w_k1_INTC_s004 + 1.1424 w_k1_KO_s004
                                 + 0.847992 w_k1_MAT_s004
                                 + 1.04318 w_k1_MSFT_s004
                                 + 1.06632 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.20229 w_k1_ADM_s004
                                 + 1.10562 w_k1_AXP_s004
                                 + 1.40149 w_k1_DIS_s004
                                 + 1.24818 w_k1_FDX_s004
                                 + 1.14202 w_k1_IBM_s004
                                 + 1.31325 w_k1_INTC_s004
                                 + 1.32522 w_k1_KO_s004 + 1.41321 w_k1_MAT_s004
                                 + 1.23634 w_k1_MSFT_s004 + 1.2559 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.17728 w_k1_ADM_s004
                                 + 1.03147 w_k1_AXP_s004
                                 + 1.00272 w_k1_DIS_s004
                                 + 0.922314 w_k1_FDX_s004
                                 + 1.51284 w_k1_IBM_s004
                                 + 1.08637 w_k1_INTC_s004
                                 + 1.31574 w_k1_KO_s004 + 1.23327 w_k1_MAT_s004
                                 + 0.925876 w_k1_MSFT_s004
                                 + 1.01317 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.954205 w_k1_ADM_s004
                                 + 0.685141 w_k1_AXP_s004
                                 + 0.679481 w_k1_DIS_s004
                                 + 0.729813 w_k1_FDX_s004
                                 + 0.852466 w_k1_IBM_s004
                                 + 0.917503 w_k1_INTC_s004
                                 + 0.913561 w_k1_KO_s004
                                 + 0.904453 w_k1_MAT_s004
                                 + 0.748551 w_k1_MSFT_s004
                                 + 0.811199 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.34911 w_k1_ADM_s004
                                 + 1.29559 w_k1_AXP_s004
                                 + 1.74874 w_k1_DIS_s004
                                 + 1.12444 w_k1_FDX_s004
                                 + 1.55552 w_k1_IBM_s004
                                 + 1.44987 w_k1_INTC_s004
                                 + 1.76806 w_k1_KO_s004 + 1.4171 w_k1_MAT_s004
                                 + 1.32826 w_k1_MSFT_s004
                                 + 1.27214 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.01893 w_k1_ADM_s004
                                 + 0.848875 w_k1_AXP_s004
                                 + 0.990828 w_k1_DIS_s004
                                 + 0.93667 w_k1_FDX_s004
                                 + 1.00167 w_k1_IBM_s004
                                 + 1.08256 w_k1_INTC_s004
                                 + 1.18059 w_k1_KO_s004
                                 + 0.897032 w_k1_MAT_s004
                                 + 0.900889 w_k1_MSFT_s004
                                 + 0.955742 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.40524 w_k1_ADM_s004
                                 + 0.815985 w_k1_AXP_s004
                                 + 1.01919 w_k1_DIS_s004
                                 + 1.13647 w_k1_FDX_s004
                                 + 0.714437 w_k1_IBM_s004
                                 + 0.976992 w_k1_INTC_s004
                                 + 0.969057 w_k1_KO_s004
                                 + 0.697826 w_k1_MAT_s004
                                 + 0.989909 w_k1_MSFT_s004
                                 + 0.955842 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.719669 w_k1_ADM_s005
                                 + 1.069 w_k1_AXP_s005 + 0.977267 w_k1_DIS_s005
                                 + 0.983668 w_k1_FDX_s005
                                 + 1.6572 w_k1_IBM_s005
                                 + 1.12397 w_k1_INTC_s005
                                 + 1.29645 w_k1_KO_s005 + 1.72518 w_k1_MAT_s005
                                 + 1.04786 w_k1_MSFT_s005
                                 + 1.21098 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.945442 w_k1_ADM_s005
                                 + 0.986925 w_k1_AXP_s005
                                 + 0.819082 w_k1_DIS_s005
                                 + 0.938988 w_k1_FDX_s005
                                 + 0.904136 w_k1_IBM_s005
                                 + 1.03406 w_k1_INTC_s005
                                 + 0.925488 w_k1_KO_s005
                                 + 1.23484 w_k1_MAT_s005
                                 + 0.92498 w_k1_MSFT_s005
                                 + 1.02602 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.797992 w_k1_ADM_s005
                                 + 0.89677 w_k1_AXP_s005
                                 + 0.908291 w_k1_DIS_s005
                                 + 0.836618 w_k1_FDX_s005
                                 + 1.27925 w_k1_IBM_s005
                                 + 1.04586 w_k1_INTC_s005
                                 + 1.03967 w_k1_KO_s005 + 1.09621 w_k1_MAT_s005
                                 + 1.04528 w_k1_MSFT_s005
                                 + 0.842695 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.09302 w_k1_ADM_s005
                                 + 1.23944 w_k1_AXP_s005
                                 + 1.49996 w_k1_DIS_s005
                                 + 1.25952 w_k1_FDX_s005
                                 + 1.45188 w_k1_IBM_s005
                                 + 1.3889 w_k1_INTC_s005 + 1.55741 w_k1_KO_s005
                                 + 1.71867 w_k1_MAT_s005
                                 + 1.2951 w_k1_MSFT_s005 + 1.25157 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.727581 w_k1_ADM_s005
                                 + 0.654949 w_k1_AXP_s005
                                 + 0.684313 w_k1_DIS_s005
                                 + 0.755912 w_k1_FDX_s005
                                 + 0.99972 w_k1_IBM_s005
                                 + 0.884517 w_k1_INTC_s005
                                 + 0.870647 w_k1_KO_s005
                                 + 0.955969 w_k1_MAT_s005
                                 + 0.816281 w_k1_MSFT_s005
                                 + 0.890866 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.11668 w_k1_ADM_s005
                                 + 1.55096 w_k1_AXP_s005
                                 + 1.00008 w_k1_DIS_s005
                                 + 1.12055 w_k1_FDX_s005
                                 + 1.04726 w_k1_IBM_s005
                                 + 1.5674 w_k1_INTC_s005 + 1.13501 w_k1_KO_s005
                                 + 1.22084 w_k1_MAT_s005
                                 + 1.17582 w_k1_MSFT_s005
                                 + 1.13723 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00581 w_k1_ADM_s005
                                 + 0.882939 w_k1_AXP_s005
                                 + 1.13808 w_k1_DIS_s005
                                 + 1.08697 w_k1_FDX_s005
                                 + 0.918261 w_k1_IBM_s005
                                 + 1.19002 w_k1_INTC_s005
                                 + 1.04645 w_k1_KO_s005 + 1.13191 w_k1_MAT_s005
                                 + 1.21572 w_k1_MSFT_s005
                                 + 0.978652 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.880722 w_k1_ADM_s005
                                 + 0.606704 w_k1_AXP_s005
                                 + 0.77171 w_k1_DIS_s005
                                 + 0.853552 w_k1_FDX_s005
                                 + 0.695883 w_k1_IBM_s005
                                 + 0.960786 w_k1_INTC_s005
                                 + 0.748531 w_k1_KO_s005
                                 + 0.501939 w_k1_MAT_s005
                                 + 0.872469 w_k1_MSFT_s005
                                 + 0.838697 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.42959 w_k1_ADM_s006
                                 + 1.33565 w_k1_AXP_s006
                                 + 1.13832 w_k1_DIS_s006
                                 + 1.10131 w_k1_FDX_s006 + 1.0324 w_k1_IBM_s006
                                 + 0.894782 w_k1_INTC_s006
                                 + 1.25192 w_k1_KO_s006 + 1.50599 w_k1_MAT_s006
                                 + 0.883006 w_k1_MSFT_s006
                                 + 1.21095 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.25593 w_k1_ADM_s006
                                 + 1.25662 w_k1_AXP_s006
                                 + 1.30237 w_k1_DIS_s006
                                 + 1.14553 w_k1_FDX_s006 + 1.2653 w_k1_IBM_s006
                                 + 0.960276 w_k1_INTC_s006
                                 + 1.23398 w_k1_KO_s006 + 1.76909 w_k1_MAT_s006
                                 + 1.04591 w_k1_MSFT_s006
                                 + 1.09969 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.877355 w_k1_ADM_s006
                                 + 0.743055 w_k1_AXP_s006
                                 + 0.71681 w_k1_DIS_s006
                                 + 0.761967 w_k1_FDX_s006
                                 + 0.960198 w_k1_IBM_s006
                                 + 1.0444 w_k1_INTC_s006 + 1.00643 w_k1_KO_s006
                                 + 1.0178 w_k1_MAT_s006
                                 + 0.899798 w_k1_MSFT_s006
                                 + 0.863893 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.29313 w_k1_ADM_s006
                                 + 1.11577 w_k1_AXP_s006
                                 + 1.27865 w_k1_DIS_s006 + 1.2545 w_k1_FDX_s006
                                 + 0.765862 w_k1_IBM_s006
                                 + 1.0478 w_k1_INTC_s006 + 1.00829 w_k1_KO_s006
                                 + 0.752356 w_k1_MAT_s006
                                 + 1.161 w_k1_MSFT_s006 + 1.07061 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.13587 w_k1_ADM_s006
                                 + 0.869697 w_k1_AXP_s006
                                 + 1.01865 w_k1_DIS_s006
                                 + 0.941389 w_k1_FDX_s006
                                 + 0.747159 w_k1_IBM_s006
                                 + 0.906774 w_k1_INTC_s006
                                 + 0.898923 w_k1_KO_s006
                                 + 0.729077 w_k1_MAT_s006
                                 + 0.921463 w_k1_MSFT_s006
                                 + 0.979247 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.05619 w_k1_ADM_s006
                                 + 1.52999 w_k1_AXP_s006
                                 + 1.69683 w_k1_DIS_s006
                                 + 1.27696 w_k1_FDX_s006
                                 + 1.56034 w_k1_IBM_s006
                                 + 1.38221 w_k1_INTC_s006
                                 + 1.68933 w_k1_KO_s006 + 2.42592 w_k1_MAT_s006
                                 + 1.25461 w_k1_MSFT_s006
                                 + 1.22939 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.2718 w_k1_ADM_s006
                                 + 0.999331 w_k1_AXP_s006
                                 + 0.903643 w_k1_DIS_s006
                                 + 0.944052 w_k1_FDX_s006
                                 + 1.03408 w_k1_IBM_s006
                                 + 0.946375 w_k1_INTC_s006
                                 + 1.10402 w_k1_KO_s006 + 1.22239 w_k1_MAT_s006
                                 + 1.05374 w_k1_MSFT_s006
                                 + 1.00088 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.10734 w_k1_ADM_s006
                                 + 0.914642 w_k1_AXP_s006
                                 + 1.0504 w_k1_DIS_s006
                                 + 0.961911 w_k1_FDX_s006
                                 + 1.20241 w_k1_IBM_s006
                                 + 1.11963 w_k1_INTC_s006
                                 + 1.19612 w_k1_KO_s006 + 1.35787 w_k1_MAT_s006
                                 + 1.11807 w_k1_MSFT_s006
                                 + 0.981968 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.68502 w_k1_ADM_s007
                                 + 0.872393 w_k1_AXP_s007
                                 + 0.738264 w_k1_DIS_s007
                                 + 0.773623 w_k1_FDX_s007
                                 + 0.935912 w_k1_IBM_s007
                                 + 0.759046 w_k1_INTC_s007
                                 + 0.906679 w_k1_KO_s007 + 1.2785 w_k1_MAT_s007
                                 + 0.82496 w_k1_MSFT_s007
                                 + 0.835247 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.34359 w_k1_ADM_s007
                                 + 1.56244 w_k1_AXP_s007
                                 + 1.67261 w_k1_DIS_s007
                                 + 1.27013 w_k1_FDX_s007
                                 + 1.30816 w_k1_IBM_s007
                                 + 1.34276 w_k1_INTC_s007
                                 + 1.23719 w_k1_KO_s007 + 1.40954 w_k1_MAT_s007
                                 + 1.20967 w_k1_MSFT_s007
                                 + 1.14748 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.73473 w_k1_ADM_s007
                                 + 0.63318 w_k1_AXP_s007
                                 + 0.839138 w_k1_DIS_s007
                                 + 0.684372 w_k1_FDX_s007
                                 + 1.1571 w_k1_IBM_s007
                                 + 0.679471 w_k1_INTC_s007
                                 + 0.982113 w_k1_KO_s007
                                 + 1.16231 w_k1_MAT_s007
                                 + 0.861222 w_k1_MSFT_s007
                                 + 0.902548 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.02885 w_k1_ADM_s007
                                 + 0.845596 w_k1_AXP_s007
                                 + 0.937809 w_k1_DIS_s007
                                 + 0.946253 w_k1_FDX_s007
                                 + 0.656449 w_k1_IBM_s007
                                 + 1.02589 w_k1_INTC_s007
                                 + 0.849626 w_k1_KO_s007
                                 + 0.747455 w_k1_MAT_s007
                                 + 0.95606 w_k1_MSFT_s007
                                 + 0.831781 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.731693 w_k1_ADM_s007
                                 + 0.706442 w_k1_AXP_s007
                                 + 0.621767 w_k1_DIS_s007
                                 + 0.718359 w_k1_FDX_s007
                                 + 0.693298 w_k1_IBM_s007
                                 + 0.85095 w_k1_INTC_s007
                                 + 0.731613 w_k1_KO_s007
                                 + 0.594413 w_k1_MAT_s007
                                 + 0.729312 w_k1_MSFT_s007
                                 + 0.775351 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.914438 w_k1_ADM_s007
                                 + 1.15 w_k1_AXP_s007 + 0.95497 w_k1_DIS_s007
                                 + 0.844582 w_k1_FDX_s007
                                 + 1.30104 w_k1_IBM_s007
                                 + 0.973814 w_k1_INTC_s007
                                 + 1.05948 w_k1_KO_s007 + 2.06266 w_k1_MAT_s007
                                 + 0.959675 w_k1_MSFT_s007
                                 + 1.06685 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.19651 w_k1_ADM_s007
                                 + 1.12045 w_k1_AXP_s007 + 1.0412 w_k1_DIS_s007
                                 + 1.09277 w_k1_FDX_s007
                                 + 0.867568 w_k1_IBM_s007
                                 + 1.10255 w_k1_INTC_s007
                                 + 0.891068 w_k1_KO_s007
                                 + 1.24343 w_k1_MAT_s007
                                 + 1.09449 w_k1_MSFT_s007
                                 + 0.998875 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.939096 w_k1_ADM_s007
                                 + 0.789225 w_k1_AXP_s007
                                 + 1.47678 w_k1_DIS_s007
                                 + 0.993254 w_k1_FDX_s007
                                 + 1.28012 w_k1_IBM_s007
                                 + 0.984346 w_k1_INTC_s007
                                 + 1.4316 w_k1_KO_s007 + 1.27463 w_k1_MAT_s007
                                 + 1.18378 w_k1_MSFT_s007
                                 + 1.05174 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.713267 w_k1_ADM_s010
                                 + 0.674772 w_k1_AXP_s010
                                 + 0.799231 w_k1_DIS_s010
                                 + 0.798741 w_k1_FDX_s010
                                 + 0.930604 w_k1_IBM_s010
                                 + 0.977534 w_k1_INTC_s010
                                 + 0.951558 w_k1_KO_s010
                                 + 0.659889 w_k1_MAT_s010
                                 + 0.954455 w_k1_MSFT_s010
                                 + 0.918552 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.948551 w_k1_ADM_s010
                                 + 0.760619 w_k1_AXP_s010
                                 + 0.817338 w_k1_DIS_s010
                                 + 0.914595 w_k1_FDX_s010
                                 + 0.684885 w_k1_IBM_s010
                                 + 0.873241 w_k1_INTC_s010
                                 + 0.723798 w_k1_KO_s010
                                 + 0.673873 w_k1_MAT_s010
                                 + 0.916892 w_k1_MSFT_s010
                                 + 0.884832 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.17966 w_k1_ADM_s010
                                 + 1.08629 w_k1_AXP_s010 + 1.0249 w_k1_DIS_s010
                                 + 1.11631 w_k1_FDX_s010
                                 + 0.801789 w_k1_IBM_s010
                                 + 1.06837 w_k1_INTC_s010
                                 + 0.994628 w_k1_KO_s010
                                 + 0.744928 w_k1_MAT_s010
                                 + 1.29204 w_k1_MSFT_s010 + 1.0377 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.21747 w_k1_ADM_s010
                                 + 1.19178 w_k1_AXP_s010
                                 + 1.55619 w_k1_DIS_s010
                                 + 1.36962 w_k1_FDX_s010
                                 + 0.938973 w_k1_IBM_s010
                                 + 1.46211 w_k1_INTC_s010
                                 + 1.12983 w_k1_KO_s010
                                 + 0.945986 w_k1_MAT_s010
                                 + 1.51196 w_k1_MSFT_s010
                                 + 1.09384 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.803838 w_k1_ADM_s010
                                 + 1.17334 w_k1_AXP_s010 + 1.2109 w_k1_DIS_s010
                                 + 1.12084 w_k1_FDX_s010
                                 + 1.23981 w_k1_IBM_s010
                                 + 1.11096 w_k1_INTC_s010
                                 + 1.20933 w_k1_KO_s010 + 1.17394 w_k1_MAT_s010
                                 + 1.16934 w_k1_MSFT_s010
                                 + 1.03198 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.19653 w_k1_ADM_s010
                                 + 1.2842 w_k1_AXP_s010 + 1.32834 w_k1_DIS_s010
                                 + 1.21668 w_k1_FDX_s010 + 1.5835 w_k1_IBM_s010
                                 + 1.26221 w_k1_INTC_s010
                                 + 1.31962 w_k1_KO_s010 + 1.40113 w_k1_MAT_s010
                                 + 1.27866 w_k1_MSFT_s010
                                 + 1.16935 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.902038 w_k1_ADM_s010
                                 + 0.907647 w_k1_AXP_s010
                                 + 0.75902 w_k1_DIS_s010 + 0.8824 w_k1_FDX_s010
                                 + 1.13777 w_k1_IBM_s010
                                 + 0.984732 w_k1_INTC_s010
                                 + 0.91104 w_k1_KO_s010
                                 + 0.828566 w_k1_MAT_s010
                                 + 0.890966 w_k1_MSFT_s010
                                 + 0.947498 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02061 w_k1_ADM_s010
                                 + 1.15578 w_k1_AXP_s010
                                 + 1.12382 w_k1_DIS_s010
                                 + 1.10207 w_k1_FDX_s010
                                 + 1.02507 w_k1_IBM_s010
                                 + 0.938583 w_k1_INTC_s010
                                 + 0.919526 w_k1_KO_s010
                                 + 1.04622 w_k1_MAT_s010
                                 + 1.07674 w_k1_MSFT_s010
                                 + 1.00259 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.885957 w_k1_ADM_s011
                                 + 0.62172 w_k1_AXP_s011
                                 + 0.988586 w_k1_DIS_s011
                                 + 0.907655 w_k1_FDX_s011
                                 + 0.737868 w_k1_IBM_s011
                                 + 1.10579 w_k1_INTC_s011
                                 + 0.816806 w_k1_KO_s011
                                 + 0.568581 w_k1_MAT_s011
                                 + 0.966059 w_k1_MSFT_s011
                                 + 0.82549 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.16021 w_k1_ADM_s011
                                 + 1.28281 w_k1_AXP_s011
                                 + 1.54484 w_k1_DIS_s011
                                 + 1.46372 w_k1_FDX_s011
                                 + 0.819327 w_k1_IBM_s011
                                 + 1.03524 w_k1_INTC_s011
                                 + 1.03522 w_k1_KO_s011
                                 + 0.998295 w_k1_MAT_s011
                                 + 1.27135 w_k1_MSFT_s011
                                 + 1.02897 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.00235 w_k1_ADM_s011
                                 + 1.0486 w_k1_AXP_s011
                                 + 0.952259 w_k1_DIS_s011
                                 + 1.1677 w_k1_FDX_s011
                                 + 0.705824 w_k1_IBM_s011
                                 + 1.0786 w_k1_INTC_s011
                                 + 0.793589 w_k1_KO_s011
                                 + 0.797738 w_k1_MAT_s011
                                 + 0.965685 w_k1_MSFT_s011
                                 + 0.949621 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.910647 w_k1_ADM_s011
                                 + 0.984953 w_k1_AXP_s011
                                 + 0.97434 w_k1_DIS_s011 + 1.0483 w_k1_FDX_s011
                                 + 0.940619 w_k1_IBM_s011
                                 + 1.01666 w_k1_INTC_s011
                                 + 0.93044 w_k1_KO_s011 + 1.11426 w_k1_MAT_s011
                                 + 1.10051 w_k1_MSFT_s011
                                 + 0.89467 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.731667 w_k1_ADM_s011
                                 + 0.706646 w_k1_AXP_s011
                                 + 0.643493 w_k1_DIS_s011
                                 + 0.730948 w_k1_FDX_s011
                                 + 1.07001 w_k1_IBM_s011
                                 + 0.722048 w_k1_INTC_s011
                                 + 0.865179 w_k1_KO_s011
                                 + 1.14276 w_k1_MAT_s011
                                 + 0.764209 w_k1_MSFT_s011
                                 + 0.848267 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.850316 w_k1_ADM_s011
                                 + 0.725751 w_k1_AXP_s011
                                 + 0.653428 w_k1_DIS_s011
                                 + 0.875688 w_k1_FDX_s011
                                 + 0.652756 w_k1_IBM_s011
                                 + 0.870876 w_k1_INTC_s011
                                 + 0.708515 w_k1_KO_s011
                                 + 0.922791 w_k1_MAT_s011
                                 + 0.914143 w_k1_MSFT_s011
                                 + 0.817888 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.25986 w_k1_ADM_s011
                                 + 1.29004 w_k1_AXP_s011
                                 + 1.53615 w_k1_DIS_s011
                                 + 1.33232 w_k1_FDX_s011
                                 + 1.20334 w_k1_IBM_s011
                                 + 1.11037 w_k1_INTC_s011
                                 + 1.34713 w_k1_KO_s011 + 1.34676 w_k1_MAT_s011
                                 + 1.21302 w_k1_MSFT_s011
                                 + 1.07122 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.84011 w_k1_ADM_s011
                                 + 1.16363 w_k1_AXP_s011 + 1.0194 w_k1_DIS_s011
                                 + 0.962052 w_k1_FDX_s011
                                 + 1.41876 w_k1_IBM_s011
                                 + 0.938972 w_k1_INTC_s011
                                 + 1.13767 w_k1_KO_s011 + 1.81665 w_k1_MAT_s011
                                 + 0.945144 w_k1_MSFT_s011
                                 + 1.00041 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.994238 w_k1_ADM_s012
                                 + 0.740115 w_k1_AXP_s012
                                 + 0.747132 w_k1_DIS_s012
                                 + 0.935158 w_k1_FDX_s012
                                 + 0.594492 w_k1_IBM_s012
                                 + 0.916102 w_k1_INTC_s012
                                 + 0.741571 w_k1_KO_s012
                                 + 0.405722 w_k1_MAT_s012
                                 + 0.901737 w_k1_MSFT_s012
                                 + 0.862363 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.02521 w_k1_ADM_s012
                                 + 1.13333 w_k1_AXP_s012
                                 + 1.45528 w_k1_DIS_s012
                                 + 1.00498 w_k1_FDX_s012
                                 + 1.67347 w_k1_IBM_s012
                                 + 1.03565 w_k1_INTC_s012 + 1.5289 w_k1_KO_s012
                                 + 1.49411 w_k1_MAT_s012
                                 + 1.08203 w_k1_MSFT_s012
                                 + 1.12998 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.00072 w_k1_ADM_s012
                                 + 0.923605 w_k1_AXP_s012
                                 + 0.926118 w_k1_DIS_s012
                                 + 0.922647 w_k1_FDX_s012
                                 + 1.0557 w_k1_IBM_s012
                                 + 1.15718 w_k1_INTC_s012
                                 + 1.04991 w_k1_KO_s012
                                 + 0.598988 w_k1_MAT_s012
                                 + 0.959872 w_k1_MSFT_s012
                                 + 0.927447 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.00253 w_k1_ADM_s012
                                 + 0.792581 w_k1_AXP_s012
                                 + 0.857456 w_k1_DIS_s012
                                 + 0.777192 w_k1_FDX_s012
                                 + 1.28124 w_k1_IBM_s012
                                 + 0.913847 w_k1_INTC_s012
                                 + 1.19939 w_k1_KO_s012
                                 + 0.824396 w_k1_MAT_s012
                                 + 0.95446 w_k1_MSFT_s012
                                 + 0.944299 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.28175 w_k1_ADM_s012
                                 + 1.25556 w_k1_AXP_s012
                                 + 1.06539 w_k1_DIS_s012
                                 + 1.19648 w_k1_FDX_s012
                                 + 0.781181 w_k1_IBM_s012
                                 + 1.09169 w_k1_INTC_s012
                                 + 0.939113 w_k1_KO_s012
                                 + 0.713604 w_k1_MAT_s012
                                 + 1.0918 w_k1_MSFT_s012
                                 + 0.923311 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.944435 w_k1_ADM_s012
                                 + 1.01246 w_k1_AXP_s012
                                 + 0.745764 w_k1_DIS_s012
                                 + 0.841006 w_k1_FDX_s012
                                 + 0.799764 w_k1_IBM_s012
                                 + 0.928588 w_k1_INTC_s012
                                 + 0.840424 w_k1_KO_s012
                                 + 0.753145 w_k1_MAT_s012
                                 + 0.857876 w_k1_MSFT_s012
                                 + 0.841617 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.11267 w_k1_ADM_s012
                                 + 1.18251 w_k1_AXP_s012 + 1.25 w_k1_DIS_s012
                                 + 1.15657 w_k1_FDX_s012
                                 + 1.05902 w_k1_IBM_s012
                                 + 1.12395 w_k1_INTC_s012
                                 + 1.09475 w_k1_KO_s012 + 1.09875 w_k1_MAT_s012
                                 + 1.2089 w_k1_MSFT_s012 + 1.0012 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.32519 w_k1_ADM_s012
                                 + 1.40346 w_k1_AXP_s012
                                 + 1.52354 w_k1_DIS_s012
                                 + 1.36597 w_k1_FDX_s012
                                 + 1.38766 w_k1_IBM_s012
                                 + 1.33309 w_k1_INTC_s012
                                 + 1.29377 w_k1_KO_s012
                                 + 0.930106 w_k1_MAT_s012
                                 + 1.30525 w_k1_MSFT_s012
                                 + 1.13992 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.65507 w_k1_ADM_s013
                                 + 1.65379 w_k1_AXP_s013
                                 + 1.75602 w_k1_DIS_s013 + 1.4086 w_k1_FDX_s013
                                 + 1.52382 w_k1_IBM_s013
                                 + 1.29086 w_k1_INTC_s013
                                 + 1.46146 w_k1_KO_s013 + 1.53802 w_k1_MAT_s013
                                 + 1.17607 w_k1_MSFT_s013
                                 + 1.06407 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.14352 w_k1_ADM_s013
                                 + 1.04072 w_k1_AXP_s013
                                 + 1.24627 w_k1_DIS_s013
                                 + 0.975551 w_k1_FDX_s013
                                 + 1.2571 w_k1_IBM_s013
                                 + 0.985217 w_k1_INTC_s013
                                 + 1.38365 w_k1_KO_s013 + 1.13999 w_k1_MAT_s013
                                 + 1.1313 w_k1_MSFT_s013 + 0.9409 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.5251 w_k1_ADM_s013
                                 + 0.91467 w_k1_AXP_s013
                                 + 1.23094 w_k1_DIS_s013
                                 + 1.29616 w_k1_FDX_s013
                                 + 0.800199 w_k1_IBM_s013
                                 + 1.1322 w_k1_INTC_s013 + 1.09367 w_k1_KO_s013
                                 + 0.575587 w_k1_MAT_s013
                                 + 1.293 w_k1_MSFT_s013 + 0.999792 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.967969 w_k1_ADM_s013
                                 + 0.801153 w_k1_AXP_s013
                                 + 0.726165 w_k1_DIS_s013
                                 + 0.881549 w_k1_FDX_s013
                                 + 0.588112 w_k1_IBM_s013
                                 + 0.891032 w_k1_INTC_s013
                                 + 0.662809 w_k1_KO_s013
                                 + 0.484292 w_k1_MAT_s013
                                 + 0.798911 w_k1_MSFT_s013
                                 + 0.688448 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15343 w_k1_ADM_s013
                                 + 0.794577 w_k1_AXP_s013
                                 + 0.954734 w_k1_DIS_s013
                                 + 1.03702 w_k1_FDX_s013
                                 + 0.78107 w_k1_IBM_s013
                                 + 0.900725 w_k1_INTC_s013
                                 + 0.777369 w_k1_KO_s013
                                 + 0.567921 w_k1_MAT_s013
                                 + 1.03385 w_k1_MSFT_s013
                                 + 0.922309 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.39355 w_k1_ADM_s013
                                 + 1.25657 w_k1_AXP_s013 + 1.0775 w_k1_DIS_s013
                                 + 1.16292 w_k1_FDX_s013
                                 + 0.884806 w_k1_IBM_s013
                                 + 1.41203 w_k1_INTC_s013
                                 + 0.987966 w_k1_KO_s013
                                 + 1.06566 w_k1_MAT_s013
                                 + 1.1484 w_k1_MSFT_s013 + 0.94981 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.1431 w_k1_ADM_s013
                                 + 0.794329 w_k1_AXP_s013
                                 + 0.685223 w_k1_DIS_s013
                                 + 0.782549 w_k1_FDX_s013
                                 + 1.00895 w_k1_IBM_s013
                                 + 0.650921 w_k1_INTC_s013
                                 + 0.987997 w_k1_KO_s013
                                 + 0.921805 w_k1_MAT_s013
                                 + 0.799056 w_k1_MSFT_s013
                                 + 0.95028 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.985169 w_k1_ADM_s013
                                 + 1.24891 w_k1_AXP_s013
                                 + 1.04196 w_k1_DIS_s013
                                 + 0.987597 w_k1_FDX_s013
                                 + 1.68405 w_k1_IBM_s013
                                 + 1.11383 w_k1_INTC_s013
                                 + 1.24281 w_k1_KO_s013 + 1.703 w_k1_MAT_s013
                                 + 1.10275 w_k1_MSFT_s013
                                 + 0.981762 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.09434 w_k1_ADM_s014
                                 + 1.03215 w_k1_AXP_s014
                                 + 1.40429 w_k1_DIS_s014
                                 + 1.08169 w_k1_FDX_s014
                                 + 1.08233 w_k1_IBM_s014
                                 + 1.02919 w_k1_INTC_s014
                                 + 1.02208 w_k1_KO_s014
                                 + 0.913778 w_k1_MAT_s014
                                 + 1.16132 w_k1_MSFT_s014
                                 + 1.17428 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.690512 w_k1_ADM_s014
                                 + 0.748719 w_k1_AXP_s014
                                 + 0.663426 w_k1_DIS_s014
                                 + 0.742383 w_k1_FDX_s014
                                 + 0.896015 w_k1_IBM_s014
                                 + 0.89721 w_k1_INTC_s014
                                 + 0.788879 w_k1_KO_s014
                                 + 0.76947 w_k1_MAT_s014
                                 + 0.776571 w_k1_MSFT_s014
                                 + 0.802459 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.902551 w_k1_ADM_s014
                                 + 0.786815 w_k1_AXP_s014
                                 + 0.981024 w_k1_DIS_s014
                                 + 0.929801 w_k1_FDX_s014
                                 + 0.666086 w_k1_IBM_s014
                                 + 0.9854 w_k1_INTC_s014
                                 + 0.870825 w_k1_KO_s014
                                 + 0.722731 w_k1_MAT_s014
                                 + 0.955215 w_k1_MSFT_s014
                                 + 0.898474 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.16666 w_k1_ADM_s014
                                 + 1.07742 w_k1_AXP_s014
                                 + 1.01399 w_k1_DIS_s014
                                 + 1.16869 w_k1_FDX_s014
                                 + 0.593494 w_k1_IBM_s014
                                 + 0.952078 w_k1_INTC_s014
                                 + 0.797558 w_k1_KO_s014
                                 + 0.604316 w_k1_MAT_s014
                                 + 1.14299 w_k1_MSFT_s014
                                 + 1.09878 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.886348 w_k1_ADM_s014
                                 + 0.692974 w_k1_AXP_s014
                                 + 0.833989 w_k1_DIS_s014
                                 + 0.822269 w_k1_FDX_s014
                                 + 0.8532 w_k1_IBM_s014
                                 + 0.812078 w_k1_INTC_s014
                                 + 0.832741 w_k1_KO_s014
                                 + 0.785284 w_k1_MAT_s014
                                 + 0.908614 w_k1_MSFT_s014
                                 + 0.952317 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.970059 w_k1_ADM_s014
                                 + 1.18988 w_k1_AXP_s014
                                 + 1.09548 w_k1_DIS_s014
                                 + 0.981176 w_k1_FDX_s014
                                 + 1.32759 w_k1_IBM_s014
                                 + 0.926916 w_k1_INTC_s014
                                 + 1.01666 w_k1_KO_s014 + 1.32082 w_k1_MAT_s014
                                 + 0.868637 w_k1_MSFT_s014
                                 + 0.968717 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.92055 w_k1_ADM_s014
                                 + 1.06097 w_k1_AXP_s014
                                 + 0.82931 w_k1_DIS_s014
                                 + 1.02068 w_k1_FDX_s014
                                 + 0.807554 w_k1_IBM_s014
                                 + 0.882604 w_k1_INTC_s014
                                 + 0.793575 w_k1_KO_s014
                                 + 0.746583 w_k1_MAT_s014
                                 + 0.78271 w_k1_MSFT_s014
                                 + 0.910166 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.04293 w_k1_ADM_s014
                                 + 1.64039 w_k1_AXP_s014 + 1.453 w_k1_DIS_s014
                                 + 1.28803 w_k1_FDX_s014
                                 + 1.07727 w_k1_IBM_s014
                                 + 1.27524 w_k1_INTC_s014
                                 + 1.20208 w_k1_KO_s014 + 1.30514 w_k1_MAT_s014
                                 + 1.14571 w_k1_MSFT_s014
                                 + 1.16127 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.861425 w_k1_ADM_s015
                                 + 0.845912 w_k1_AXP_s015
                                 + 0.87458 w_k1_DIS_s015
                                 + 0.869356 w_k1_FDX_s015
                                 + 0.908002 w_k1_IBM_s015
                                 + 1.04645 w_k1_INTC_s015
                                 + 0.93719 w_k1_KO_s015
                                 + 0.691151 w_k1_MAT_s015
                                 + 1.03979 w_k1_MSFT_s015
                                 + 0.878919 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.842351 w_k1_ADM_s015
                                 + 1.08782 w_k1_AXP_s015
                                 + 1.38333 w_k1_DIS_s015
                                 + 1.02219 w_k1_FDX_s015
                                 + 1.53282 w_k1_IBM_s015
                                 + 1.35292 w_k1_INTC_s015
                                 + 1.41879 w_k1_KO_s015 + 1.35177 w_k1_MAT_s015
                                 + 1.1617 w_k1_MSFT_s015 + 1.07294 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.779325 w_k1_ADM_s015
                                 + 0.633345 w_k1_AXP_s015
                                 + 0.637412 w_k1_DIS_s015
                                 + 0.770732 w_k1_FDX_s015
                                 + 0.611064 w_k1_IBM_s015
                                 + 0.830875 w_k1_INTC_s015
                                 + 0.651136 w_k1_KO_s015
                                 + 0.439997 w_k1_MAT_s015
                                 + 0.764358 w_k1_MSFT_s015
                                 + 0.822423 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.984211 w_k1_ADM_s015
                                 + 0.967297 w_k1_AXP_s015
                                 + 1.12848 w_k1_DIS_s015 + 1.1 w_k1_FDX_s015
                                 + 0.957175 w_k1_IBM_s015
                                 + 1.20029 w_k1_INTC_s015
                                 + 0.997523 w_k1_KO_s015
                                 + 0.894591 w_k1_MAT_s015
                                 + 1.09518 w_k1_MSFT_s015
                                 + 0.903958 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.37923 w_k1_ADM_s015
                                 + 1.40684 w_k1_AXP_s015
                                 + 1.75951 w_k1_DIS_s015
                                 + 1.50745 w_k1_FDX_s015 + 1.1583 w_k1_IBM_s015
                                 + 1.36281 w_k1_INTC_s015
                                 + 1.43028 w_k1_KO_s015 + 1.19812 w_k1_MAT_s015
                                 + 1.51785 w_k1_MSFT_s015
                                 + 1.18034 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.974476 w_k1_ADM_s015
                                 + 1.4421 w_k1_AXP_s015 + 1.3105 w_k1_DIS_s015
                                 + 1.22757 w_k1_FDX_s015 + 1.3395 w_k1_IBM_s015
                                 + 1.17963 w_k1_INTC_s015
                                 + 1.17214 w_k1_KO_s015 + 1.54262 w_k1_MAT_s015
                                 + 1.20747 w_k1_MSFT_s015
                                 + 1.10421 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.750044 w_k1_ADM_s015
                                 + 0.759419 w_k1_AXP_s015
                                 + 0.61045 w_k1_DIS_s015
                                 + 0.712576 w_k1_FDX_s015
                                 + 0.9701 w_k1_IBM_s015
                                 + 0.879819 w_k1_INTC_s015
                                 + 0.73426 w_k1_KO_s015
                                 + 0.729824 w_k1_MAT_s015
                                 + 0.769024 w_k1_MSFT_s015
                                 + 0.776017 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.940667 w_k1_ADM_s015
                                 + 0.977142 w_k1_AXP_s015
                                 + 0.868199 w_k1_DIS_s015
                                 + 0.905011 w_k1_FDX_s015
                                 + 1.40556 w_k1_IBM_s015
                                 + 0.85149 w_k1_INTC_s015
                                 + 1.08485 w_k1_KO_s015 + 1.44132 w_k1_MAT_s015
                                 + 0.92015 w_k1_MSFT_s015
                                 + 0.963691 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.02535 w_k1_ADM_s016
                                 + 0.862928 w_k1_AXP_s016
                                 + 0.750967 w_k1_DIS_s016
                                 + 0.869819 w_k1_FDX_s016
                                 + 0.939374 w_k1_IBM_s016
                                 + 0.869436 w_k1_INTC_s016
                                 + 0.901015 w_k1_KO_s016
                                 + 1.05652 w_k1_MAT_s016
                                 + 0.915336 w_k1_MSFT_s016
                                 + 0.970249 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.17023 w_k1_ADM_s016
                                 + 0.828616 w_k1_AXP_s016
                                 + 0.879518 w_k1_DIS_s016
                                 + 1.10454 w_k1_FDX_s016
                                 + 0.67777 w_k1_IBM_s016
                                 + 1.04042 w_k1_INTC_s016
                                 + 0.907401 w_k1_KO_s016
                                 + 0.726521 w_k1_MAT_s016
                                 + 0.965131 w_k1_MSFT_s016
                                 + 0.946613 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.21616 w_k1_ADM_s016
                                 + 0.96171 w_k1_AXP_s016
                                 + 0.766701 w_k1_DIS_s016
                                 + 0.882605 w_k1_FDX_s016
                                 + 0.993443 w_k1_IBM_s016
                                 + 0.912069 w_k1_INTC_s016
                                 + 0.84303 w_k1_KO_s016 + 1.07418 w_k1_MAT_s016
                                 + 0.863016 w_k1_MSFT_s016
                                 + 0.858187 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.0306 w_k1_ADM_s016
                                 + 1.00826 w_k1_AXP_s016
                                 + 1.06331 w_k1_DIS_s016
                                 + 0.978079 w_k1_FDX_s016
                                 + 1.27744 w_k1_IBM_s016
                                 + 0.896101 w_k1_INTC_s016
                                 + 1.14016 w_k1_KO_s016 + 1.34887 w_k1_MAT_s016
                                 + 1.03418 w_k1_MSFT_s016
                                 + 1.06507 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.3996 w_k1_ADM_s016
                                 + 1.5139 w_k1_AXP_s016 + 1.48337 w_k1_DIS_s016
                                 + 1.44389 w_k1_FDX_s016
                                 + 0.968978 w_k1_IBM_s016
                                 + 1.26099 w_k1_INTC_s016
                                 + 1.04389 w_k1_KO_s016 + 1.03971 w_k1_MAT_s016
                                 + 1.39343 w_k1_MSFT_s016
                                 + 1.18414 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.14675 w_k1_ADM_s016
                                 + 1.09157 w_k1_AXP_s016
                                 + 1.49692 w_k1_DIS_s016
                                 + 1.17894 w_k1_FDX_s016
                                 + 1.19919 w_k1_IBM_s016
                                 + 1.16369 w_k1_INTC_s016
                                 + 1.23337 w_k1_KO_s016 + 1.31499 w_k1_MAT_s016
                                 + 1.31347 w_k1_MSFT_s016
                                 + 1.11613 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.97998 w_k1_ADM_s016
                                 + 0.694459 w_k1_AXP_s016
                                 + 0.853961 w_k1_DIS_s016
                                 + 0.942513 w_k1_FDX_s016
                                 + 0.708231 w_k1_IBM_s016
                                 + 1.00386 w_k1_INTC_s016
                                 + 0.848104 w_k1_KO_s016
                                 + 0.706339 w_k1_MAT_s016
                                 + 0.999859 w_k1_MSFT_s016
                                 + 0.912036 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.23126 w_k1_ADM_s016
                                 + 1.3695 w_k1_AXP_s016 + 1.45148 w_k1_DIS_s016
                                 + 1.21137 w_k1_FDX_s016
                                 + 1.67411 w_k1_IBM_s016
                                 + 1.08493 w_k1_INTC_s016
                                 + 1.32894 w_k1_KO_s016 + 1.91923 w_k1_MAT_s016
                                 + 1.13701 w_k1_MSFT_s016
                                 + 1.08334 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.975445 w_k1_ADM_s017
                                 + 0.889182 w_k1_AXP_s017
                                 + 0.906944 w_k1_DIS_s017
                                 + 0.953051 w_k1_FDX_s017
                                 + 0.591857 w_k1_IBM_s017
                                 + 1.03622 w_k1_INTC_s017
                                 + 0.685734 w_k1_KO_s017
                                 + 0.711707 w_k1_MAT_s017
                                 + 0.937432 w_k1_MSFT_s017
                                 + 0.872683 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.722019 w_k1_ADM_s017
                                 + 0.595655 w_k1_AXP_s017
                                 + 0.567416 w_k1_DIS_s017
                                 + 0.697671 w_k1_FDX_s017
                                 + 0.651549 w_k1_IBM_s017
                                 + 0.94891 w_k1_INTC_s017
                                 + 0.583297 w_k1_KO_s017
                                 + 0.441628 w_k1_MAT_s017
                                 + 0.729044 w_k1_MSFT_s017
                                 + 0.720434 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.24621 w_k1_ADM_s017
                                 + 1.44482 w_k1_AXP_s017
                                 + 1.57984 w_k1_DIS_s017
                                 + 1.28146 w_k1_FDX_s017
                                 + 1.24984 w_k1_IBM_s017
                                 + 1.20264 w_k1_INTC_s017
                                 + 1.06472 w_k1_KO_s017
                                 + 0.985696 w_k1_MAT_s017
                                 + 1.35533 w_k1_MSFT_s017
                                 + 1.02419 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.965956 w_k1_ADM_s017
                                 + 1.14471 w_k1_AXP_s017
                                 + 0.974802 w_k1_DIS_s017
                                 + 0.847618 w_k1_FDX_s017
                                 + 1.2888 w_k1_IBM_s017
                                 + 0.899796 w_k1_INTC_s017
                                 + 0.939679 w_k1_KO_s017
                                 + 1.12134 w_k1_MAT_s017
                                 + 0.889177 w_k1_MSFT_s017
                                 + 1.04025 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0873 w_k1_ADM_s017
                                 + 1.3633 w_k1_AXP_s017 + 1.21713 w_k1_DIS_s017
                                 + 1.1931 w_k1_FDX_s017
                                 + 0.867459 w_k1_IBM_s017
                                 + 1.11897 w_k1_INTC_s017
                                 + 0.965169 w_k1_KO_s017
                                 + 0.902201 w_k1_MAT_s017
                                 + 1.0984 w_k1_MSFT_s017 + 1.02865 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.92741 w_k1_ADM_s017
                                 + 0.977705 w_k1_AXP_s017
                                 + 0.914524 w_k1_DIS_s017
                                 + 0.926592 w_k1_FDX_s017
                                 + 0.888092 w_k1_IBM_s017
                                 + 1.23219 w_k1_INTC_s017
                                 + 0.853305 w_k1_KO_s017
                                 + 0.798312 w_k1_MAT_s017
                                 + 1.02675 w_k1_MSFT_s017
                                 + 0.934575 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.822886 w_k1_ADM_s017
                                 + 1.12863 w_k1_AXP_s017
                                 + 1.02956 w_k1_DIS_s017
                                 + 0.877365 w_k1_FDX_s017
                                 + 1.61079 w_k1_IBM_s017
                                 + 1.25628 w_k1_INTC_s017
                                 + 1.24924 w_k1_KO_s017 + 1.48784 w_k1_MAT_s017
                                 + 1.05032 w_k1_MSFT_s017
                                 + 0.95317 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.737955 w_k1_ADM_s017
                                 + 0.730642 w_k1_AXP_s017
                                 + 0.658049 w_k1_DIS_s017
                                 + 0.663928 w_k1_FDX_s017
                                 + 1.14055 w_k1_IBM_s017
                                 + 0.70007 w_k1_INTC_s017
                                 + 0.78095 w_k1_KO_s017
                                 + 0.848184 w_k1_MAT_s017
                                 + 0.614732 w_k1_MSFT_s017
                                 + 0.790718 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.46849 w_k1_ADM_s100
                                 + 1.69053 w_k1_AXP_s100
                                 + 1.49939 w_k1_DIS_s100
                                 + 1.14129 w_k1_FDX_s100
                                 + 1.74747 w_k1_IBM_s100
                                 + 1.07774 w_k1_INTC_s100
                                 + 1.49426 w_k1_KO_s100 + 1.79474 w_k1_MAT_s100
                                 + 1.21362 w_k1_MSFT_s100 + 1.3703 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.22476 w_k1_ADM_s100
                                 + 1.53636 w_k1_AXP_s100
                                 + 1.43574 w_k1_DIS_s100
                                 + 1.39843 w_k1_FDX_s100
                                 + 1.21123 w_k1_IBM_s100
                                 + 1.35168 w_k1_INTC_s100
                                 + 1.12895 w_k1_KO_s100 + 1.27658 w_k1_MAT_s100
                                 + 1.17636 w_k1_MSFT_s100
                                 + 1.01143 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.890646 w_k1_ADM_s100
                                 + 1.16267 w_k1_AXP_s100
                                 + 0.879619 w_k1_DIS_s100
                                 + 1.04005 w_k1_FDX_s100
                                 + 0.960127 w_k1_IBM_s100
                                 + 0.889706 w_k1_INTC_s100
                                 + 0.941152 w_k1_KO_s100
                                 + 1.03153 w_k1_MAT_s100
                                 + 0.872524 w_k1_MSFT_s100
                                 + 0.875188 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.30793 w_k1_ADM_s100
                                 + 1.40987 w_k1_AXP_s100
                                 + 1.23807 w_k1_DIS_s100
                                 + 1.36865 w_k1_FDX_s100
                                 + 0.91228 w_k1_IBM_s100
                                 + 0.97874 w_k1_INTC_s100
                                 + 1.00258 w_k1_KO_s100
                                 + 0.993824 w_k1_MAT_s100
                                 + 1.05079 w_k1_MSFT_s100 + 1.0707 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.810355 w_k1_ADM_s100
                                 + 0.859975 w_k1_AXP_s100
                                 + 0.572162 w_k1_DIS_s100
                                 + 0.740364 w_k1_FDX_s100
                                 + 0.744211 w_k1_IBM_s100
                                 + 0.870783 w_k1_INTC_s100
                                 + 0.665838 w_k1_KO_s100
                                 + 0.706967 w_k1_MAT_s100
                                 + 0.775953 w_k1_MSFT_s100
                                 + 0.83405 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.06897 w_k1_ADM_s100
                                 + 1.24934 w_k1_AXP_s100
                                 + 0.929719 w_k1_DIS_s100
                                 + 1.06966 w_k1_FDX_s100
                                 + 0.736374 w_k1_IBM_s100
                                 + 1.10968 w_k1_INTC_s100
                                 + 0.876727 w_k1_KO_s100
                                 + 0.705105 w_k1_MAT_s100
                                 + 0.97845 w_k1_MSFT_s100
                                 + 0.982093 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.88886 w_k1_ADM_s100
                                 + 0.733445 w_k1_AXP_s100
                                 + 0.839541 w_k1_DIS_s100
                                 + 0.817578 w_k1_FDX_s100
                                 + 1.14539 w_k1_IBM_s100
                                 + 0.961095 w_k1_INTC_s100
                                 + 0.991213 w_k1_KO_s100
                                 + 0.996084 w_k1_MAT_s100
                                 + 0.928815 w_k1_MSFT_s100
                                 + 0.91449 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.888063 w_k1_ADM_s100
                                 + 1.30467 w_k1_AXP_s100
                                 + 1.39537 w_k1_DIS_s100
                                 + 1.10537 w_k1_FDX_s100
                                 + 1.33139 w_k1_IBM_s100
                                 + 1.28792 w_k1_INTC_s100 + 1.1688 w_k1_KO_s100
                                 + 1.05363 w_k1_MAT_s100
                                 + 1.05828 w_k1_MSFT_s100
                                 + 1.07752 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.2545 w_k1_ADM_s101
                                 + 1.15951 w_k1_AXP_s101
                                 + 1.50214 w_k1_DIS_s101
                                 + 1.51973 w_k1_FDX_s101
                                 + 0.760075 w_k1_IBM_s101
                                 + 1.26722 w_k1_INTC_s101
                                 + 1.06545 w_k1_KO_s101
                                 + 0.564202 w_k1_MAT_s101
                                 + 1.54316 w_k1_MSFT_s101
                                 + 1.16545 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.18988 w_k1_ADM_s101
                                 + 1.22345 w_k1_AXP_s101
                                 + 1.49785 w_k1_DIS_s101 + 1.2754 w_k1_FDX_s101
                                 + 1.53723 w_k1_IBM_s101
                                 + 1.35836 w_k1_INTC_s101
                                 + 1.52076 w_k1_KO_s101 + 1.01471 w_k1_MAT_s101
                                 + 1.40489 w_k1_MSFT_s101
                                 + 1.15883 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.915291 w_k1_ADM_s101
                                 + 0.820201 w_k1_AXP_s101
                                 + 0.874622 w_k1_DIS_s101
                                 + 0.786887 w_k1_FDX_s101
                                 + 1.33985 w_k1_IBM_s101
                                 + 0.71281 w_k1_INTC_s101
                                 + 1.06178 w_k1_KO_s101
                                 + 0.923308 w_k1_MAT_s101
                                 + 0.811544 w_k1_MSFT_s101
                                 + 0.94988 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.20047 w_k1_ADM_s101
                                 + 0.861362 w_k1_AXP_s101
                                 + 0.967983 w_k1_DIS_s101
                                 + 0.932818 w_k1_FDX_s101
                                 + 1.07862 w_k1_IBM_s101
                                 + 0.962911 w_k1_INTC_s101
                                 + 0.958674 w_k1_KO_s101
                                 + 0.511136 w_k1_MAT_s101
                                 + 0.909409 w_k1_MSFT_s101
                                 + 0.942047 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.852765 w_k1_ADM_s101
                                 + 0.538484 w_k1_AXP_s101
                                 + 0.419902 w_k1_DIS_s101
                                 + 0.605631 w_k1_FDX_s101
                                 + 0.856488 w_k1_IBM_s101
                                 + 0.729542 w_k1_INTC_s101
                                 + 0.670755 w_k1_KO_s101
                                 + 0.414962 w_k1_MAT_s101
                                 + 0.618285 w_k1_MSFT_s101
                                 + 0.6572 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.239 w_k1_ADM_s101
                                 + 1.13533 w_k1_AXP_s101
                                 + 1.09354 w_k1_DIS_s101
                                 + 1.20452 w_k1_FDX_s101
                                 + 0.98345 w_k1_IBM_s101
                                 + 1.06974 w_k1_INTC_s101
                                 + 1.15482 w_k1_KO_s101
                                 + 0.997749 w_k1_MAT_s101
                                 + 1.09977 w_k1_MSFT_s101
                                 + 1.07194 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.13926 w_k1_ADM_s101
                                 + 0.751534 w_k1_AXP_s101
                                 + 0.828597 w_k1_DIS_s101
                                 + 1.06312 w_k1_FDX_s101
                                 + 0.740007 w_k1_IBM_s101
                                 + 0.891655 w_k1_INTC_s101
                                 + 0.900542 w_k1_KO_s101
                                 + 0.480429 w_k1_MAT_s101
                                 + 0.900733 w_k1_MSFT_s101
                                 + 0.920081 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.858747 w_k1_ADM_s101
                                 + 0.620441 w_k1_AXP_s101
                                 + 0.635942 w_k1_DIS_s101
                                 + 0.788792 w_k1_FDX_s101
                                 + 0.878228 w_k1_IBM_s101
                                 + 0.699592 w_k1_INTC_s101
                                 + 0.714434 w_k1_KO_s101
                                 + 0.662505 w_k1_MAT_s101
                                 + 0.791215 w_k1_MSFT_s101
                                 + 0.844479 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.27895 w_k1_ADM_s102
                                 + 0.900914 w_k1_AXP_s102
                                 + 1.10117 w_k1_DIS_s102
                                 + 0.993717 w_k1_FDX_s102
                                 + 1.11477 w_k1_IBM_s102
                                 + 0.798993 w_k1_INTC_s102
                                 + 1.08502 w_k1_KO_s102 + 1.41252 w_k1_MAT_s102
                                 + 1.02926 w_k1_MSFT_s102
                                 + 0.974381 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.37917 w_k1_ADM_s102
                                 + 1.25744 w_k1_AXP_s102 + 1.3145 w_k1_DIS_s102
                                 + 1.26713 w_k1_FDX_s102
                                 + 1.39627 w_k1_IBM_s102
                                 + 1.07153 w_k1_INTC_s102
                                 + 1.31957 w_k1_KO_s102 + 1.33216 w_k1_MAT_s102
                                 + 1.21313 w_k1_MSFT_s102
                                 + 1.31917 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.870436 w_k1_ADM_s102
                                 + 0.736188 w_k1_AXP_s102
                                 + 0.749822 w_k1_DIS_s102
                                 + 0.813208 w_k1_FDX_s102
                                 + 0.868506 w_k1_IBM_s102
                                 + 0.904793 w_k1_INTC_s102
                                 + 0.954597 w_k1_KO_s102
                                 + 0.971898 w_k1_MAT_s102
                                 + 0.934446 w_k1_MSFT_s102
                                 + 0.948659 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.10418 w_k1_ADM_s102
                                 + 0.704706 w_k1_AXP_s102
                                 + 0.855191 w_k1_DIS_s102
                                 + 0.842742 w_k1_FDX_s102
                                 + 1.07742 w_k1_IBM_s102
                                 + 0.942292 w_k1_INTC_s102
                                 + 1.15054 w_k1_KO_s102 + 1.11236 w_k1_MAT_s102
                                 + 1.03038 w_k1_MSFT_s102
                                 + 0.975863 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.39815 w_k1_ADM_s102
                                 + 1.14057 w_k1_AXP_s102
                                 + 1.19853 w_k1_DIS_s102
                                 + 1.23186 w_k1_FDX_s102
                                 + 0.88403 w_k1_IBM_s102
                                 + 0.956414 w_k1_INTC_s102
                                 + 1.15211 w_k1_KO_s102 + 1.06899 w_k1_MAT_s102
                                 + 1.17227 w_k1_MSFT_s102
                                 + 1.01834 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.80988 w_k1_ADM_s102
                                 + 0.60712 w_k1_AXP_s102
                                 + 0.726487 w_k1_DIS_s102
                                 + 0.75625 w_k1_FDX_s102
                                 + 0.944453 w_k1_IBM_s102
                                 + 0.773045 w_k1_INTC_s102
                                 + 0.934365 w_k1_KO_s102
                                 + 0.81716 w_k1_MAT_s102
                                 + 0.915706 w_k1_MSFT_s102
                                 + 0.889442 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.0083 w_k1_ADM_s102
                                 + 0.633062 w_k1_AXP_s102
                                 + 0.890981 w_k1_DIS_s102
                                 + 0.932854 w_k1_FDX_s102
                                 + 0.721357 w_k1_IBM_s102
                                 + 0.910946 w_k1_INTC_s102
                                 + 0.937357 w_k1_KO_s102 + 0.7594 w_k1_MAT_s102
                                 + 1.02561 w_k1_MSFT_s102
                                 + 0.913335 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.05418 w_k1_ADM_s102
                                 + 0.798918 w_k1_AXP_s102
                                 + 0.988299 w_k1_DIS_s102
                                 + 1.10654 w_k1_FDX_s102
                                 + 0.823638 w_k1_IBM_s102
                                 + 1.20353 w_k1_INTC_s102
                                 + 1.02063 w_k1_KO_s102
                                 + 0.584856 w_k1_MAT_s102
                                 + 1.1016 w_k1_MSFT_s102
                                 + 0.999799 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.30022 w_k1_ADM_s103
                                 + 1.79239 w_k1_AXP_s103
                                 + 1.64539 w_k1_DIS_s103
                                 + 1.41832 w_k1_FDX_s103
                                 + 0.88042 w_k1_IBM_s103
                                 + 1.01674 w_k1_INTC_s103
                                 + 1.16175 w_k1_KO_s103 + 1.14496 w_k1_MAT_s103
                                 + 1.34668 w_k1_MSFT_s103 + 1.1003 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.06547 w_k1_ADM_s103
                                 + 1.3825 w_k1_AXP_s103
                                 + 0.997192 w_k1_DIS_s103
                                 + 0.940709 w_k1_FDX_s103
                                 + 1.34732 w_k1_IBM_s103
                                 + 1.05219 w_k1_INTC_s103
                                 + 1.07066 w_k1_KO_s103
                                 + 0.985098 w_k1_MAT_s103
                                 + 0.928486 w_k1_MSFT_s103
                                 + 1.07162 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.16778 w_k1_ADM_s103
                                 + 1.63827 w_k1_AXP_s103 + 1.703 w_k1_DIS_s103
                                 + 1.38316 w_k1_FDX_s103
                                 + 1.30635 w_k1_IBM_s103
                                 + 1.34193 w_k1_INTC_s103
                                 + 1.15105 w_k1_KO_s103 + 1.1471 w_k1_MAT_s103
                                 + 1.3127 w_k1_MSFT_s103 + 1.22372 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.19421 w_k1_ADM_s103
                                 + 1.27307 w_k1_AXP_s103
                                 + 1.22805 w_k1_DIS_s103
                                 + 1.06758 w_k1_FDX_s103
                                 + 1.07478 w_k1_IBM_s103
                                 + 1.08235 w_k1_INTC_s103
                                 + 1.16601 w_k1_KO_s103
                                 + 0.951354 w_k1_MAT_s103
                                 + 1.27148 w_k1_MSFT_s103
                                 + 1.03018 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.767999 w_k1_ADM_s103
                                 + 0.873879 w_k1_AXP_s103
                                 + 0.860716 w_k1_DIS_s103
                                 + 0.927885 w_k1_FDX_s103
                                 + 0.67673 w_k1_IBM_s103
                                 + 0.811559 w_k1_INTC_s103
                                 + 0.891324 w_k1_KO_s103
                                 + 0.571767 w_k1_MAT_s103
                                 + 0.947753 w_k1_MSFT_s103
                                 + 0.860158 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.34445 w_k1_ADM_s103
                                 + 1.55478 w_k1_AXP_s103
                                 + 1.44428 w_k1_DIS_s103
                                 + 1.14636 w_k1_FDX_s103
                                 + 1.71872 w_k1_IBM_s103
                                 + 0.866831 w_k1_INTC_s103
                                 + 1.48879 w_k1_KO_s103 + 1.60495 w_k1_MAT_s103
                                 + 0.982888 w_k1_MSFT_s103
                                 + 1.27813 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.832932 w_k1_ADM_s103
                                 + 0.864678 w_k1_AXP_s103
                                 + 0.678127 w_k1_DIS_s103
                                 + 0.755309 w_k1_FDX_s103
                                 + 0.955386 w_k1_IBM_s103
                                 + 0.837626 w_k1_INTC_s103
                                 + 0.865762 w_k1_KO_s103
                                 + 0.836628 w_k1_MAT_s103
                                 + 0.832176 w_k1_MSFT_s103
                                 + 0.90428 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.24696 w_k1_ADM_s103
                                 + 0.963562 w_k1_AXP_s103
                                 + 1.08061 w_k1_DIS_s103
                                 + 1.03604 w_k1_FDX_s103
                                 + 0.857539 w_k1_IBM_s103
                                 + 0.985476 w_k1_INTC_s103
                                 + 0.993118 w_k1_KO_s103
                                 + 0.679309 w_k1_MAT_s103
                                 + 1.05917 w_k1_MSFT_s103
                                 + 1.03907 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.847421 w_k1_ADM_s104
                                 + 0.973258 w_k1_AXP_s104
                                 + 1.09767 w_k1_DIS_s104
                                 + 0.975808 w_k1_FDX_s104
                                 + 0.953952 w_k1_IBM_s104
                                 + 0.800129 w_k1_INTC_s104
                                 + 0.857706 w_k1_KO_s104
                                 + 0.841478 w_k1_MAT_s104
                                 + 0.927521 w_k1_MSFT_s104
                                 + 0.881805 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.15711 w_k1_ADM_s104
                                 + 1.16693 w_k1_AXP_s104
                                 + 1.49531 w_k1_DIS_s104
                                 + 1.18777 w_k1_FDX_s104
                                 + 1.01725 w_k1_IBM_s104
                                 + 1.10378 w_k1_INTC_s104
                                 + 1.29484 w_k1_KO_s104 + 1.17871 w_k1_MAT_s104
                                 + 1.25727 w_k1_MSFT_s104
                                 + 1.18439 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.886683 w_k1_ADM_s104
                                 + 1.11286 w_k1_AXP_s104
                                 + 1.27818 w_k1_DIS_s104
                                 + 0.914382 w_k1_FDX_s104
                                 + 1.56595 w_k1_IBM_s104
                                 + 1.05621 w_k1_INTC_s104
                                 + 1.30753 w_k1_KO_s104 + 1.02692 w_k1_MAT_s104
                                 + 1.1108 w_k1_MSFT_s104 + 1.02275 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.782362 w_k1_ADM_s104
                                 + 0.781653 w_k1_AXP_s104
                                 + 0.784068 w_k1_DIS_s104
                                 + 0.762091 w_k1_FDX_s104
                                 + 1.10341 w_k1_IBM_s104
                                 + 0.627561 w_k1_INTC_s104
                                 + 0.952911 w_k1_KO_s104
                                 + 1.29584 w_k1_MAT_s104
                                 + 0.819529 w_k1_MSFT_s104
                                 + 0.890169 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.25017 w_k1_ADM_s104
                                 + 1.91058 w_k1_AXP_s104
                                 + 1.82741 w_k1_DIS_s104
                                 + 1.55012 w_k1_FDX_s104
                                 + 1.31693 w_k1_IBM_s104
                                 + 1.65327 w_k1_INTC_s104
                                 + 1.52857 w_k1_KO_s104 + 1.58627 w_k1_MAT_s104
                                 + 1.3901 w_k1_MSFT_s104 + 1.18603 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.977162 w_k1_ADM_s104
                                 + 1.24779 w_k1_AXP_s104
                                 + 0.93048 w_k1_DIS_s104
                                 + 1.01159 w_k1_FDX_s104
                                 + 1.26472 w_k1_IBM_s104
                                 + 1.09359 w_k1_INTC_s104 + 1.068 w_k1_KO_s104
                                 + 1.31496 w_k1_MAT_s104
                                 + 1.07647 w_k1_MSFT_s104
                                 + 1.06099 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.981359 w_k1_ADM_s104
                                 + 0.979383 w_k1_AXP_s104
                                 + 0.859128 w_k1_DIS_s104
                                 + 0.983405 w_k1_FDX_s104
                                 + 0.856912 w_k1_IBM_s104
                                 + 0.947348 w_k1_INTC_s104
                                 + 0.84235 w_k1_KO_s104
                                 + 0.806165 w_k1_MAT_s104
                                 + 0.908478 w_k1_MSFT_s104
                                 + 0.806603 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.980637 w_k1_ADM_s104
                                 + 1.63537 w_k1_AXP_s104
                                 + 1.66432 w_k1_DIS_s104 + 1.2908 w_k1_FDX_s104
                                 + 1.72764 w_k1_IBM_s104
                                 + 1.04139 w_k1_INTC_s104
                                 + 1.47688 w_k1_KO_s104 + 2.40365 w_k1_MAT_s104
                                 + 1.26934 w_k1_MSFT_s104
                                 + 1.11473 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.15384 w_k1_ADM_s105
                                 + 1.24574 w_k1_AXP_s105
                                 + 1.25051 w_k1_DIS_s105
                                 + 1.10309 w_k1_FDX_s105
                                 + 1.52816 w_k1_IBM_s105
                                 + 1.22088 w_k1_INTC_s105
                                 + 1.41361 w_k1_KO_s105 + 1.50878 w_k1_MAT_s105
                                 + 1.334 w_k1_MSFT_s105 + 1.07205 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.847 w_k1_ADM_s105
                                 + 1.00044 w_k1_AXP_s105
                                 + 0.694987 w_k1_DIS_s105
                                 + 0.941458 w_k1_FDX_s105
                                 + 0.795769 w_k1_IBM_s105
                                 + 0.958691 w_k1_INTC_s105
                                 + 0.823647 w_k1_KO_s105
                                 + 0.806051 w_k1_MAT_s105
                                 + 0.836098 w_k1_MSFT_s105
                                 + 0.885717 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.640763 w_k1_ADM_s105
                                 + 0.884793 w_k1_AXP_s105
                                 + 0.9194 w_k1_DIS_s105
                                 + 0.799219 w_k1_FDX_s105
                                 + 1.20765 w_k1_IBM_s105
                                 + 1.04016 w_k1_INTC_s105
                                 + 1.00157 w_k1_KO_s105
                                 + 0.997816 w_k1_MAT_s105
                                 + 0.941512 w_k1_MSFT_s105
                                 + 0.885473 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.986057 w_k1_ADM_s105
                                 + 1.05563 w_k1_AXP_s105
                                 + 1.19256 w_k1_DIS_s105
                                 + 1.12457 w_k1_FDX_s105
                                 + 0.835102 w_k1_IBM_s105
                                 + 1.19471 w_k1_INTC_s105
                                 + 1.03771 w_k1_KO_s105
                                 + 0.701425 w_k1_MAT_s105
                                 + 1.269 w_k1_MSFT_s105 + 1.0247 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.960203 w_k1_ADM_s105
                                 + 1.28269 w_k1_AXP_s105 + 1.0735 w_k1_DIS_s105
                                 + 1.10604 w_k1_FDX_s105
                                 + 1.10591 w_k1_IBM_s105
                                 + 1.19186 w_k1_INTC_s105
                                 + 1.10629 w_k1_KO_s105 + 0.69899 w_k1_MAT_s105
                                 + 1.17566 w_k1_MSFT_s105 + 1.0648 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.763423 w_k1_ADM_s105
                                 + 0.7025 w_k1_AXP_s105
                                 + 0.888021 w_k1_DIS_s105
                                 + 0.896569 w_k1_FDX_s105
                                 + 0.898108 w_k1_IBM_s105
                                 + 0.873807 w_k1_INTC_s105
                                 + 0.9817 w_k1_KO_s105 + 0.737662 w_k1_MAT_s105
                                 + 0.974788 w_k1_MSFT_s105
                                 + 0.946512 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.30561 w_k1_ADM_s105
                                 + 1.28158 w_k1_AXP_s105
                                 + 1.53774 w_k1_DIS_s105
                                 + 1.40276 w_k1_FDX_s105
                                 + 1.19449 w_k1_IBM_s105 + 1.075 w_k1_INTC_s105
                                 + 1.19706 w_k1_KO_s105
                                 + 0.906689 w_k1_MAT_s105
                                 + 1.28188 w_k1_MSFT_s105
                                 + 1.11493 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.590655 w_k1_ADM_s105
                                 + 0.542707 w_k1_AXP_s105
                                 + 0.570181 w_k1_DIS_s105
                                 + 0.604387 w_k1_FDX_s105
                                 + 0.930099 w_k1_IBM_s105
                                 + 0.6628 w_k1_INTC_s105
                                 + 0.810017 w_k1_KO_s105
                                 + 0.792994 w_k1_MAT_s105
                                 + 0.66653 w_k1_MSFT_s105
                                 + 0.775802 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.979282 w_k1_ADM_s106
                                 + 1.03727 w_k1_AXP_s106
                                 + 1.23378 w_k1_DIS_s106 + 1.0908 w_k1_FDX_s106
                                 + 1.51372 w_k1_IBM_s106
                                 + 1.14923 w_k1_INTC_s106
                                 + 1.28498 w_k1_KO_s106 + 1.4888 w_k1_MAT_s106
                                 + 1.13227 w_k1_MSFT_s106
                                 + 1.15352 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.01348 w_k1_ADM_s106
                                 + 1.05504 w_k1_AXP_s106
                                 + 1.37564 w_k1_DIS_s106
                                 + 1.29435 w_k1_FDX_s106
                                 + 0.778349 w_k1_IBM_s106
                                 + 1.1497 w_k1_INTC_s106 + 1.0475 w_k1_KO_s106
                                 + 0.912404 w_k1_MAT_s106
                                 + 1.12398 w_k1_MSFT_s106
                                 + 1.00608 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.00662 w_k1_ADM_s106
                                 + 0.795691 w_k1_AXP_s106
                                 + 0.780718 w_k1_DIS_s106
                                 + 0.994398 w_k1_FDX_s106
                                 + 0.575757 w_k1_IBM_s106
                                 + 0.976642 w_k1_INTC_s106
                                 + 0.794177 w_k1_KO_s106
                                 + 0.626291 w_k1_MAT_s106
                                 + 1.00854 w_k1_MSFT_s106
                                 + 0.933168 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.982999 w_k1_ADM_s106
                                 + 0.970276 w_k1_AXP_s106
                                 + 0.96012 w_k1_DIS_s106
                                 + 0.98272 w_k1_FDX_s106
                                 + 0.98986 w_k1_IBM_s106
                                 + 1.17502 w_k1_INTC_s106
                                 + 1.01775 w_k1_KO_s106
                                 + 0.958726 w_k1_MAT_s106
                                 + 1.03055 w_k1_MSFT_s106
                                 + 0.936253 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.995071 w_k1_ADM_s106
                                 + 0.969059 w_k1_AXP_s106
                                 + 1.20262 w_k1_DIS_s106
                                 + 1.15554 w_k1_FDX_s106
                                 + 1.00116 w_k1_IBM_s106
                                 + 1.06254 w_k1_INTC_s106
                                 + 1.04917 w_k1_KO_s106 + 1.01866 w_k1_MAT_s106
                                 + 1.02938 w_k1_MSFT_s106
                                 + 1.11663 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.995342 w_k1_ADM_s106
                                 + 0.881596 w_k1_AXP_s106
                                 + 0.797279 w_k1_DIS_s106
                                 + 0.891835 w_k1_FDX_s106
                                 + 0.898665 w_k1_IBM_s106
                                 + 0.950228 w_k1_INTC_s106
                                 + 0.912734 w_k1_KO_s106
                                 + 0.850617 w_k1_MAT_s106
                                 + 0.855605 w_k1_MSFT_s106
                                 + 1.00809 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.18122 w_k1_ADM_s106
                                 + 1.36576 w_k1_AXP_s106
                                 + 1.54513 w_k1_DIS_s106
                                 + 1.42293 w_k1_FDX_s106
                                 + 1.12613 w_k1_IBM_s106
                                 + 1.32616 w_k1_INTC_s106
                                 + 1.23687 w_k1_KO_s106 + 1.11799 w_k1_MAT_s106
                                 + 1.17287 w_k1_MSFT_s106
                                 + 1.18044 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.84529 w_k1_ADM_s106
                                 + 0.703821 w_k1_AXP_s106
                                 + 0.687623 w_k1_DIS_s106
                                 + 0.750808 w_k1_FDX_s106
                                 + 1.09482 w_k1_IBM_s106
                                 + 0.908657 w_k1_INTC_s106
                                 + 0.980263 w_k1_KO_s106
                                 + 1.14632 w_k1_MAT_s106
                                 + 0.811604 w_k1_MSFT_s106
                                 + 0.916534 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.0375 w_k1_ADM_s107
                                 + 1.20516 w_k1_AXP_s107
                                 + 1.00528 w_k1_DIS_s107
                                 + 1.06527 w_k1_FDX_s107
                                 + 1.17908 w_k1_IBM_s107
                                 + 1.28962 w_k1_INTC_s107
                                 + 1.04324 w_k1_KO_s107 + 1.40357 w_k1_MAT_s107
                                 + 1.0364 w_k1_MSFT_s107 + 1.01128 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.739173 w_k1_ADM_s107
                                 + 0.870475 w_k1_AXP_s107
                                 + 0.858382 w_k1_DIS_s107
                                 + 0.842473 w_k1_FDX_s107
                                 + 0.818628 w_k1_IBM_s107
                                 + 0.946444 w_k1_INTC_s107
                                 + 0.807743 w_k1_KO_s107
                                 + 0.903738 w_k1_MAT_s107
                                 + 0.857723 w_k1_MSFT_s107
                                 + 0.829827 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.3865 w_k1_ADM_s107
                                 + 1.21419 w_k1_AXP_s107
                                 + 1.17408 w_k1_DIS_s107
                                 + 1.26992 w_k1_FDX_s107
                                 + 0.687316 w_k1_IBM_s107
                                 + 1.19254 w_k1_INTC_s107
                                 + 0.759474 w_k1_KO_s107
                                 + 0.600945 w_k1_MAT_s107
                                 + 1.14894 w_k1_MSFT_s107
                                 + 0.971686 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.674055 w_k1_ADM_s107
                                 + 0.756481 w_k1_AXP_s107
                                 + 0.679965 w_k1_DIS_s107
                                 + 0.759112 w_k1_FDX_s107
                                 + 0.84572 w_k1_IBM_s107
                                 + 0.904529 w_k1_INTC_s107
                                 + 0.729834 w_k1_KO_s107
                                 + 0.712319 w_k1_MAT_s107
                                 + 0.716313 w_k1_MSFT_s107
                                 + 0.727683 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18882 w_k1_ADM_s107
                                 + 0.852266 w_k1_AXP_s107
                                 + 1.08716 w_k1_DIS_s107
                                 + 1.04435 w_k1_FDX_s107
                                 + 1.05826 w_k1_IBM_s107
                                 + 1.24984 w_k1_INTC_s107
                                 + 1.10123 w_k1_KO_s107
                                 + 0.834081 w_k1_MAT_s107
                                 + 1.08017 w_k1_MSFT_s107
                                 + 1.03191 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.093 w_k1_ADM_s107
                                 + 1.37227 w_k1_AXP_s107
                                 + 1.50131 w_k1_DIS_s107
                                 + 1.20949 w_k1_FDX_s107
                                 + 1.38624 w_k1_IBM_s107
                                 + 1.39381 w_k1_INTC_s107
                                 + 1.24791 w_k1_KO_s107 + 1.32514 w_k1_MAT_s107
                                 + 1.31729 w_k1_MSFT_s107 + 1.1081 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.02476 w_k1_ADM_s107
                                 + 1.0612 w_k1_AXP_s107 + 0.99199 w_k1_DIS_s107
                                 + 1.04947 w_k1_FDX_s107
                                 + 0.860128 w_k1_IBM_s107
                                 + 1.0411 w_k1_INTC_s107
                                 + 0.782305 w_k1_KO_s107
                                 + 0.983052 w_k1_MAT_s107
                                 + 0.914545 w_k1_MSFT_s107
                                 + 0.94502 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.947344 w_k1_ADM_s107
                                 + 0.909284 w_k1_AXP_s107
                                 + 1.24854 w_k1_DIS_s107
                                 + 1.12873 w_k1_FDX_s107
                                 + 0.739789 w_k1_IBM_s107
                                 + 1.10253 w_k1_INTC_s107
                                 + 1.00796 w_k1_KO_s107
                                 + 0.584527 w_k1_MAT_s107
                                 + 1.22933 w_k1_MSFT_s107
                                 + 0.925632 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20175 w_k1_ADM_s110
                                 + 1.00851 w_k1_AXP_s110
                                 + 1.70537 w_k1_DIS_s110
                                 + 1.28663 w_k1_FDX_s110 + 1.4642 w_k1_IBM_s110
                                 + 1.17407 w_k1_INTC_s110
                                 + 1.45231 w_k1_KO_s110 + 1.21277 w_k1_MAT_s110
                                 + 1.37236 w_k1_MSFT_s110
                                 + 1.37504 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.02512 w_k1_ADM_s110
                                 + 1.13154 w_k1_AXP_s110
                                 + 1.32156 w_k1_DIS_s110
                                 + 1.20071 w_k1_FDX_s110
                                 + 0.952309 w_k1_IBM_s110
                                 + 1.11854 w_k1_INTC_s110
                                 + 1.24325 w_k1_KO_s110 + 1.18613 w_k1_MAT_s110
                                 + 1.23916 w_k1_MSFT_s110
                                 + 1.17673 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.861038 w_k1_ADM_s110
                                 + 0.943784 w_k1_AXP_s110
                                 + 1.15766 w_k1_DIS_s110
                                 + 0.973244 w_k1_FDX_s110
                                 + 1.41022 w_k1_IBM_s110
                                 + 1.05802 w_k1_INTC_s110 + 1.1059 w_k1_KO_s110
                                 + 1.17755 w_k1_MAT_s110
                                 + 1.00756 w_k1_MSFT_s110
                                 + 0.970207 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.740479 w_k1_ADM_s110
                                 + 0.586218 w_k1_AXP_s110
                                 + 0.592528 w_k1_DIS_s110
                                 + 0.75622 w_k1_FDX_s110
                                 + 0.720256 w_k1_IBM_s110
                                 + 0.97427 w_k1_INTC_s110
                                 + 0.801831 w_k1_KO_s110
                                 + 0.618475 w_k1_MAT_s110
                                 + 0.826207 w_k1_MSFT_s110
                                 + 0.862793 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.869559 w_k1_ADM_s110
                                 + 0.74749 w_k1_AXP_s110
                                 + 0.746843 w_k1_DIS_s110
                                 + 0.830219 w_k1_FDX_s110
                                 + 0.994187 w_k1_IBM_s110
                                 + 0.88861 w_k1_INTC_s110
                                 + 0.965802 w_k1_KO_s110
                                 + 1.41661 w_k1_MAT_s110
                                 + 0.856299 w_k1_MSFT_s110
                                 + 0.934896 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.881169 w_k1_ADM_s110
                                 + 0.722215 w_k1_AXP_s110
                                 + 0.750622 w_k1_DIS_s110
                                 + 0.847778 w_k1_FDX_s110
                                 + 0.747004 w_k1_IBM_s110
                                 + 0.997224 w_k1_INTC_s110
                                 + 0.705055 w_k1_KO_s110
                                 + 0.709641 w_k1_MAT_s110
                                 + 0.923857 w_k1_MSFT_s110
                                 + 1.01581 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.0754 w_k1_ADM_s110
                                 + 0.982912 w_k1_AXP_s110
                                 + 0.947306 w_k1_DIS_s110
                                 + 0.95982 w_k1_FDX_s110
                                 + 1.20572 w_k1_IBM_s110
                                 + 1.04307 w_k1_INTC_s110
                                 + 1.12727 w_k1_KO_s110 + 1.47016 w_k1_MAT_s110
                                 + 0.946579 w_k1_MSFT_s110
                                 + 1.06428 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.15801 w_k1_ADM_s110
                                 + 0.915035 w_k1_AXP_s110
                                 + 1.04587 w_k1_DIS_s110
                                 + 1.05547 w_k1_FDX_s110
                                 + 0.884768 w_k1_IBM_s110
                                 + 0.861225 w_k1_INTC_s110
                                 + 1.13627 w_k1_KO_s110
                                 + 0.852205 w_k1_MAT_s110
                                 + 1.04963 w_k1_MSFT_s110
                                 + 1.02465 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.87297 w_k1_ADM_s111
                                 + 0.845629 w_k1_AXP_s111
                                 + 0.776115 w_k1_DIS_s111
                                 + 1.01809 w_k1_FDX_s111
                                 + 0.655857 w_k1_IBM_s111
                                 + 1.05791 w_k1_INTC_s111
                                 + 0.783462 w_k1_KO_s111
                                 + 0.652992 w_k1_MAT_s111
                                 + 1.02906 w_k1_MSFT_s111
                                 + 0.863993 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.29201 w_k1_ADM_s111
                                 + 1.4605 w_k1_AXP_s111 + 1.21072 w_k1_DIS_s111
                                 + 1.41768 w_k1_FDX_s111
                                 + 0.975043 w_k1_IBM_s111
                                 + 1.25141 w_k1_INTC_s111
                                 + 1.10513 w_k1_KO_s111
                                 + 0.916207 w_k1_MAT_s111
                                 + 1.22211 w_k1_MSFT_s111
                                 + 1.08027 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.983602 w_k1_ADM_s111
                                 + 0.858581 w_k1_AXP_s111
                                 + 1.01598 w_k1_DIS_s111
                                 + 0.981566 w_k1_FDX_s111
                                 + 0.90192 w_k1_IBM_s111
                                 + 0.913402 w_k1_INTC_s111
                                 + 0.989656 w_k1_KO_s111
                                 + 0.710671 w_k1_MAT_s111
                                 + 0.944164 w_k1_MSFT_s111
                                 + 1.07762 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.8068 w_k1_ADM_s111
                                 + 0.965754 w_k1_AXP_s111
                                 + 1.05851 w_k1_DIS_s111
                                 + 0.997458 w_k1_FDX_s111
                                 + 1.29862 w_k1_IBM_s111
                                 + 0.940468 w_k1_INTC_s111
                                 + 1.16918 w_k1_KO_s111 + 1.33686 w_k1_MAT_s111
                                 + 1.02587 w_k1_MSFT_s111
                                 + 0.952353 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.649098 w_k1_ADM_s111
                                 + 0.700385 w_k1_AXP_s111
                                 + 0.615525 w_k1_DIS_s111
                                 + 0.696955 w_k1_FDX_s111
                                 + 0.720501 w_k1_IBM_s111
                                 + 0.80309 w_k1_INTC_s111
                                 + 0.761818 w_k1_KO_s111
                                 + 0.677515 w_k1_MAT_s111
                                 + 0.664734 w_k1_MSFT_s111
                                 + 0.757538 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.23237 w_k1_ADM_s111
                                 + 1.41974 w_k1_AXP_s111
                                 + 1.54659 w_k1_DIS_s111
                                 + 1.22415 w_k1_FDX_s111
                                 + 1.60295 w_k1_IBM_s111
                                 + 1.25274 w_k1_INTC_s111
                                 + 1.58082 w_k1_KO_s111 + 1.52759 w_k1_MAT_s111
                                 + 1.23492 w_k1_MSFT_s111
                                 + 1.25276 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.791485 w_k1_ADM_s111
                                 + 0.863392 w_k1_AXP_s111
                                 + 0.662106 w_k1_DIS_s111
                                 + 0.833835 w_k1_FDX_s111
                                 + 1.00103 w_k1_IBM_s111
                                 + 0.966136 w_k1_INTC_s111
                                 + 0.942557 w_k1_KO_s111
                                 + 0.974515 w_k1_MAT_s111
                                 + 0.856223 w_k1_MSFT_s111
                                 + 0.800671 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.00004 w_k1_ADM_s111
                                 + 1.17025 w_k1_AXP_s111
                                 + 0.979369 w_k1_DIS_s111
                                 + 1.23847 w_k1_FDX_s111
                                 + 0.87247 w_k1_IBM_s111
                                 + 1.09758 w_k1_INTC_s111
                                 + 0.856556 w_k1_KO_s111
                                 + 0.797583 w_k1_MAT_s111
                                 + 1.07353 w_k1_MSFT_s111
                                 + 0.993725 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.717462 w_k1_ADM_s112
                                 + 0.672183 w_k1_AXP_s112
                                 + 0.596263 w_k1_DIS_s112
                                 + 0.70305 w_k1_FDX_s112
                                 + 0.823716 w_k1_IBM_s112
                                 + 0.846735 w_k1_INTC_s112
                                 + 0.8448 w_k1_KO_s112 + 0.475229 w_k1_MAT_s112
                                 + 0.778094 w_k1_MSFT_s112
                                 + 0.741447 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.49739 w_k1_ADM_s112
                                 + 1.81066 w_k1_AXP_s112
                                 + 2.18287 w_k1_DIS_s112
                                 + 1.44283 w_k1_FDX_s112
                                 + 1.92585 w_k1_IBM_s112
                                 + 1.30507 w_k1_INTC_s112
                                 + 1.90271 w_k1_KO_s112 + 2.35523 w_k1_MAT_s112
                                 + 1.66845 w_k1_MSFT_s112
                                 + 1.31365 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.08211 w_k1_ADM_s112
                                 + 1.06301 w_k1_AXP_s112
                                 + 1.15855 w_k1_DIS_s112
                                 + 1.09739 w_k1_FDX_s112 + 1.3642 w_k1_IBM_s112
                                 + 1.07542 w_k1_INTC_s112
                                 + 1.20737 w_k1_KO_s112 + 1.29238 w_k1_MAT_s112
                                 + 1.17234 w_k1_MSFT_s112
                                 + 0.91931 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.09937 w_k1_ADM_s112
                                 + 0.752618 w_k1_AXP_s112
                                 + 0.739582 w_k1_DIS_s112
                                 + 0.879663 w_k1_FDX_s112
                                 + 0.835414 w_k1_IBM_s112
                                 + 0.904715 w_k1_INTC_s112
                                 + 0.923281 w_k1_KO_s112
                                 + 0.617548 w_k1_MAT_s112
                                 + 0.875922 w_k1_MSFT_s112
                                 + 0.838718 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.20584 w_k1_ADM_s112
                                 + 0.887318 w_k1_AXP_s112
                                 + 1.06967 w_k1_DIS_s112
                                 + 1.12537 w_k1_FDX_s112
                                 + 0.905553 w_k1_IBM_s112
                                 + 0.96593 w_k1_INTC_s112
                                 + 0.967882 w_k1_KO_s112
                                 + 0.66426 w_k1_MAT_s112
                                 + 1.07997 w_k1_MSFT_s112
                                 + 0.887942 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.15834 w_k1_ADM_s112
                                 + 1.59718 w_k1_AXP_s112
                                 + 1.50982 w_k1_DIS_s112
                                 + 1.43457 w_k1_FDX_s112
                                 + 1.05574 w_k1_IBM_s112
                                 + 1.4751 w_k1_INTC_s112 + 1.2668 w_k1_KO_s112
                                 + 1.15538 w_k1_MAT_s112
                                 + 1.20183 w_k1_MSFT_s112 + 1.1183 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.16358 w_k1_ADM_s112
                                 + 1.2958 w_k1_AXP_s112 + 1.72735 w_k1_DIS_s112
                                 + 1.23409 w_k1_FDX_s112
                                 + 1.28842 w_k1_IBM_s112
                                 + 1.12575 w_k1_INTC_s112
                                 + 1.57761 w_k1_KO_s112 + 1.12711 w_k1_MAT_s112
                                 + 1.21085 w_k1_MSFT_s112
                                 + 1.03717 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.986545 w_k1_ADM_s112
                                 + 1.14103 w_k1_AXP_s112
                                 + 0.984452 w_k1_DIS_s112
                                 + 0.893623 w_k1_FDX_s112
                                 + 1.40443 w_k1_IBM_s112
                                 + 0.777292 w_k1_INTC_s112
                                 + 1.32058 w_k1_KO_s112 + 1.70272 w_k1_MAT_s112
                                 + 1.01674 w_k1_MSFT_s112
                                 + 0.934193 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.08782 w_k1_ADM_s113
                                 + 1.0781 w_k1_AXP_s113 + 1.43757 w_k1_DIS_s113
                                 + 1.11003 w_k1_FDX_s113
                                 + 1.21445 w_k1_IBM_s113
                                 + 1.2179 w_k1_INTC_s113 + 1.32001 w_k1_KO_s113
                                 + 1.054 w_k1_MAT_s113 + 1.06776 w_k1_MSFT_s113
                                 + 1.03941 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.06274 w_k1_ADM_s113
                                 + 0.764194 w_k1_AXP_s113
                                 + 0.944142 w_k1_DIS_s113
                                 + 1.00496 w_k1_FDX_s113
                                 + 1.05126 w_k1_IBM_s113
                                 + 1.10837 w_k1_INTC_s113
                                 + 1.16972 w_k1_KO_s113 + 1.10758 w_k1_MAT_s113
                                 + 0.943883 w_k1_MSFT_s113
                                 + 0.916108 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.829946 w_k1_ADM_s113
                                 + 0.746043 w_k1_AXP_s113
                                 + 0.70348 w_k1_DIS_s113
                                 + 0.824408 w_k1_FDX_s113
                                 + 0.782119 w_k1_IBM_s113
                                 + 0.976625 w_k1_INTC_s113
                                 + 0.703087 w_k1_KO_s113
                                 + 0.538203 w_k1_MAT_s113
                                 + 0.781707 w_k1_MSFT_s113
                                 + 0.826451 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.4321 w_k1_ADM_s113
                                 + 1.20121 w_k1_AXP_s113
                                 + 1.37127 w_k1_DIS_s113
                                 + 1.34039 w_k1_FDX_s113
                                 + 0.929442 w_k1_IBM_s113
                                 + 1.42377 w_k1_INTC_s113
                                 + 1.35317 w_k1_KO_s113 + 1.07009 w_k1_MAT_s113
                                 + 1.26472 w_k1_MSFT_s113
                                 + 1.18674 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.21526 w_k1_ADM_s113
                                 + 0.796346 w_k1_AXP_s113
                                 + 0.933169 w_k1_DIS_s113
                                 + 1.11371 w_k1_FDX_s113
                                 + 0.663457 w_k1_IBM_s113
                                 + 0.9724 w_k1_INTC_s113
                                 + 0.911869 w_k1_KO_s113
                                 + 0.700707 w_k1_MAT_s113
                                 + 0.970779 w_k1_MSFT_s113
                                 + 0.905551 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.10104 w_k1_ADM_s113
                                 + 1.09339 w_k1_AXP_s113
                                 + 1.25307 w_k1_DIS_s113
                                 + 1.00192 w_k1_FDX_s113
                                 + 1.61315 w_k1_IBM_s113
                                 + 1.0146 w_k1_INTC_s113 + 1.53419 w_k1_KO_s113
                                 + 1.95746 w_k1_MAT_s113
                                 + 1.0432 w_k1_MSFT_s113
                                 + 0.995215 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.762099 w_k1_ADM_s113
                                 + 0.691755 w_k1_AXP_s113
                                 + 0.659926 w_k1_DIS_s113
                                 + 0.85169 w_k1_FDX_s113
                                 + 0.591239 w_k1_IBM_s113
                                 + 0.901924 w_k1_INTC_s113
                                 + 0.664289 w_k1_KO_s113
                                 + 0.631329 w_k1_MAT_s113
                                 + 0.922846 w_k1_MSFT_s113
                                 + 0.81147 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.733033 w_k1_ADM_s113
                                 + 0.670337 w_k1_AXP_s113
                                 + 0.855775 w_k1_DIS_s113
                                 + 0.775155 w_k1_FDX_s113
                                 + 0.939076 w_k1_IBM_s113
                                 + 0.967463 w_k1_INTC_s113
                                 + 0.988953 w_k1_KO_s113
                                 + 0.613336 w_k1_MAT_s113
                                 + 0.926859 w_k1_MSFT_s113
                                 + 0.868872 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.06697 w_k1_ADM_s114
                                 + 1.26857 w_k1_AXP_s114
                                 + 1.10425 w_k1_DIS_s114
                                 + 1.14269 w_k1_FDX_s114
                                 + 0.87215 w_k1_IBM_s114
                                 + 1.05086 w_k1_INTC_s114
                                 + 1.01328 w_k1_KO_s114 + 1.00245 w_k1_MAT_s114
                                 + 1.00989 w_k1_MSFT_s114
                                 + 0.971301 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.992849 w_k1_ADM_s114
                                 + 1.01557 w_k1_AXP_s114
                                 + 1.47268 w_k1_DIS_s114
                                 + 1.07962 w_k1_FDX_s114
                                 + 0.963792 w_k1_IBM_s114
                                 + 1.07549 w_k1_INTC_s114 + 1.5384 w_k1_KO_s114
                                 + 1.10572 w_k1_MAT_s114
                                 + 1.20564 w_k1_MSFT_s114
                                 + 1.00593 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.77728 w_k1_ADM_s114
                                 + 0.963705 w_k1_AXP_s114
                                 + 0.992084 w_k1_DIS_s114
                                 + 0.816569 w_k1_FDX_s114
                                 + 1.40041 w_k1_IBM_s114
                                 + 0.910445 w_k1_INTC_s114
                                 + 1.1776 w_k1_KO_s114 + 1.36972 w_k1_MAT_s114
                                 + 0.787916 w_k1_MSFT_s114
                                 + 0.994888 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.00521 w_k1_ADM_s114
                                 + 0.778643 w_k1_AXP_s114
                                 + 0.845179 w_k1_DIS_s114
                                 + 0.89835 w_k1_FDX_s114
                                 + 0.690502 w_k1_IBM_s114
                                 + 0.771662 w_k1_INTC_s114
                                 + 0.82558 w_k1_KO_s114
                                 + 0.761059 w_k1_MAT_s114
                                 + 0.860906 w_k1_MSFT_s114
                                 + 0.936131 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.04824 w_k1_ADM_s114
                                 + 0.800564 w_k1_AXP_s114
                                 + 0.979898 w_k1_DIS_s114
                                 + 0.938872 w_k1_FDX_s114
                                 + 1.04838 w_k1_IBM_s114
                                 + 0.871009 w_k1_INTC_s114
                                 + 1.08966 w_k1_KO_s114
                                 + 0.721532 w_k1_MAT_s114
                                 + 0.921921 w_k1_MSFT_s114
                                 + 0.846051 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.12728 w_k1_ADM_s114
                                 + 1.31448 w_k1_AXP_s114
                                 + 1.19658 w_k1_DIS_s114
                                 + 1.07339 w_k1_FDX_s114
                                 + 1.33515 w_k1_IBM_s114
                                 + 0.788921 w_k1_INTC_s114
                                 + 1.15457 w_k1_KO_s114 + 1.7214 w_k1_MAT_s114
                                 + 0.814888 w_k1_MSFT_s114
                                 + 1.13496 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.16119 w_k1_ADM_s114
                                 + 1.15691 w_k1_AXP_s114
                                 + 1.60485 w_k1_DIS_s114
                                 + 1.43241 w_k1_FDX_s114
                                 + 0.962953 w_k1_IBM_s114
                                 + 1.12676 w_k1_INTC_s114
                                 + 1.30195 w_k1_KO_s114 + 1.08041 w_k1_MAT_s114
                                 + 1.2817 w_k1_MSFT_s114 + 1.13947 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.30336 w_k1_ADM_s114
                                 + 1.32753 w_k1_AXP_s114
                                 + 1.56006 w_k1_DIS_s114
                                 + 1.24104 w_k1_FDX_s114 + 1.2057 w_k1_IBM_s114
                                 + 1.10838 w_k1_INTC_s114
                                 + 1.37175 w_k1_KO_s114 + 1.09234 w_k1_MAT_s114
                                 + 1.24339 w_k1_MSFT_s114
                                 + 1.08265 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.04156 w_k1_ADM_s115
                                 + 1.00014 w_k1_AXP_s115
                                 + 1.14817 w_k1_DIS_s115
                                 + 1.09769 w_k1_FDX_s115
                                 + 1.07478 w_k1_IBM_s115
                                 + 1.25443 w_k1_INTC_s115
                                 + 1.22773 w_k1_KO_s115 + 1.08475 w_k1_MAT_s115
                                 + 1.16771 w_k1_MSFT_s115
                                 + 1.11067 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.980437 w_k1_ADM_s115
                                 + 0.906423 w_k1_AXP_s115
                                 + 1.00336 w_k1_DIS_s115
                                 + 1.14161 w_k1_FDX_s115
                                 + 0.636583 w_k1_IBM_s115
                                 + 0.794057 w_k1_INTC_s115
                                 + 0.743783 w_k1_KO_s115
                                 + 0.574439 w_k1_MAT_s115
                                 + 0.993159 w_k1_MSFT_s115
                                 + 0.903951 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.963878 w_k1_ADM_s115
                                 + 0.867532 w_k1_AXP_s115
                                 + 1.05772 w_k1_DIS_s115
                                 + 0.875859 w_k1_FDX_s115
                                 + 1.45759 w_k1_IBM_s115
                                 + 1.0346 w_k1_INTC_s115 + 1.32051 w_k1_KO_s115
                                 + 1.22501 w_k1_MAT_s115
                                 + 1.05521 w_k1_MSFT_s115
                                 + 1.14673 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.91675 w_k1_ADM_s115
                                 + 1.0169 w_k1_AXP_s115
                                 + 0.853378 w_k1_DIS_s115
                                 + 0.97209 w_k1_FDX_s115
                                 + 0.782715 w_k1_IBM_s115
                                 + 1.08438 w_k1_INTC_s115
                                 + 0.905292 w_k1_KO_s115
                                 + 0.805919 w_k1_MAT_s115
                                 + 1.01641 w_k1_MSFT_s115
                                 + 0.857414 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.843706 w_k1_ADM_s115
                                 + 0.961772 w_k1_AXP_s115
                                 + 0.798203 w_k1_DIS_s115
                                 + 0.866141 w_k1_FDX_s115
                                 + 0.941031 w_k1_IBM_s115
                                 + 0.871163 w_k1_INTC_s115
                                 + 0.884757 w_k1_KO_s115
                                 + 0.899473 w_k1_MAT_s115
                                 + 0.865108 w_k1_MSFT_s115
                                 + 0.954492 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.26365 w_k1_ADM_s115
                                 + 1.10371 w_k1_AXP_s115
                                 + 1.33638 w_k1_DIS_s115
                                 + 1.18453 w_k1_FDX_s115 + 1.2243 w_k1_IBM_s115
                                 + 0.953915 w_k1_INTC_s115
                                 + 1.19709 w_k1_KO_s115 + 1.27671 w_k1_MAT_s115
                                 + 1.12472 w_k1_MSFT_s115
                                 + 1.12499 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.895822 w_k1_ADM_s115
                                 + 1.05072 w_k1_AXP_s115
                                 + 0.887514 w_k1_DIS_s115
                                 + 0.850479 w_k1_FDX_s115
                                 + 1.19439 w_k1_IBM_s115
                                 + 0.955203 w_k1_INTC_s115
                                 + 1.09195 w_k1_KO_s115 + 1.23942 w_k1_MAT_s115
                                 + 0.854135 w_k1_MSFT_s115
                                 + 0.979968 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.959351 w_k1_ADM_s115
                                 + 1.47016 w_k1_AXP_s115
                                 + 1.54237 w_k1_DIS_s115
                                 + 1.19706 w_k1_FDX_s115
                                 + 1.59671 w_k1_IBM_s115
                                 + 1.13857 w_k1_INTC_s115 + 1.2958 w_k1_KO_s115
                                 + 1.28421 w_k1_MAT_s115
                                 + 1.17189 w_k1_MSFT_s115
                                 + 1.12822 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.0378 w_k1_ADM_s116
                                 + 0.96735 w_k1_AXP_s116
                                 + 0.934835 w_k1_DIS_s116
                                 + 0.988327 w_k1_FDX_s116
                                 + 0.821357 w_k1_IBM_s116
                                 + 0.976462 w_k1_INTC_s116
                                 + 0.824658 w_k1_KO_s116
                                 + 0.727409 w_k1_MAT_s116
                                 + 0.958896 w_k1_MSFT_s116
                                 + 0.896133 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.93662 w_k1_ADM_s116
                                 + 1.29363 w_k1_AXP_s116
                                 + 1.18714 w_k1_DIS_s116
                                 + 0.972635 w_k1_FDX_s116
                                 + 1.63736 w_k1_IBM_s116
                                 + 0.985793 w_k1_INTC_s116
                                 + 1.16292 w_k1_KO_s116 + 2.21542 w_k1_MAT_s116
                                 + 0.95658 w_k1_MSFT_s116
                                 + 0.975363 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.31682 w_k1_ADM_s116
                                 + 2.02641 w_k1_AXP_s116
                                 + 2.11715 w_k1_DIS_s116
                                 + 1.50479 w_k1_FDX_s116
                                 + 1.61788 w_k1_IBM_s116
                                 + 1.13974 w_k1_INTC_s116
                                 + 1.34827 w_k1_KO_s116 + 1.61876 w_k1_MAT_s116
                                 + 1.26798 w_k1_MSFT_s116
                                 + 1.17933 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.09744 w_k1_ADM_s116
                                 + 1.04837 w_k1_AXP_s116
                                 + 0.952814 w_k1_DIS_s116
                                 + 0.887505 w_k1_FDX_s116
                                 + 1.12187 w_k1_IBM_s116
                                 + 0.829287 w_k1_INTC_s116
                                 + 0.880032 w_k1_KO_s116
                                 + 1.02362 w_k1_MAT_s116
                                 + 0.898632 w_k1_MSFT_s116
                                 + 0.939653 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.04266 w_k1_ADM_s116
                                 + 1.10249 w_k1_AXP_s116
                                 + 1.22491 w_k1_DIS_s116
                                 + 1.11483 w_k1_FDX_s116
                                 + 1.02668 w_k1_IBM_s116
                                 + 0.875325 w_k1_INTC_s116
                                 + 1.0212 w_k1_KO_s116 + 1.49046 w_k1_MAT_s116
                                 + 1.18411 w_k1_MSFT_s116
                                 + 1.01099 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.924502 w_k1_ADM_s116
                                 + 0.79019 w_k1_AXP_s116
                                 + 0.661858 w_k1_DIS_s116
                                 + 0.73709 w_k1_FDX_s116
                                 + 0.880696 w_k1_IBM_s116
                                 + 0.648453 w_k1_INTC_s116
                                 + 0.834917 w_k1_KO_s116
                                 + 0.792081 w_k1_MAT_s116
                                 + 0.802816 w_k1_MSFT_s116
                                 + 0.820864 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.13135 w_k1_ADM_s116
                                 + 1.15947 w_k1_AXP_s116
                                 + 1.70794 w_k1_DIS_s116
                                 + 1.29716 w_k1_FDX_s116
                                 + 1.10881 w_k1_IBM_s116
                                 + 1.18724 w_k1_INTC_s116
                                 + 1.20189 w_k1_KO_s116
                                 + 0.933085 w_k1_MAT_s116
                                 + 1.43695 w_k1_MSFT_s116
                                 + 1.15881 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.12005 w_k1_ADM_s116
                                 + 0.731235 w_k1_AXP_s116
                                 + 0.757283 w_k1_DIS_s116
                                 + 0.844148 w_k1_FDX_s116
                                 + 0.720326 w_k1_IBM_s116
                                 + 0.810793 w_k1_INTC_s116
                                 + 0.715328 w_k1_KO_s116
                                 + 0.606862 w_k1_MAT_s116
                                 + 0.930308 w_k1_MSFT_s116
                                 + 0.86938 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.806784 w_k1_ADM_s117
                                 + 0.96146 w_k1_AXP_s117
                                 + 1.06893 w_k1_DIS_s117
                                 + 1.09184 w_k1_FDX_s117
                                 + 0.689347 w_k1_IBM_s117
                                 + 1.04299 w_k1_INTC_s117
                                 + 0.921088 w_k1_KO_s117
                                 + 1.00412 w_k1_MAT_s117
                                 + 1.0107 w_k1_MSFT_s117 + 1.01573 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.964836 w_k1_ADM_s117
                                 + 1.13335 w_k1_AXP_s117
                                 + 1.12902 w_k1_DIS_s117 + 1.1496 w_k1_FDX_s117
                                 + 1.02042 w_k1_IBM_s117
                                 + 0.840773 w_k1_INTC_s117
                                 + 1.03146 w_k1_KO_s117
                                 + 0.906911 w_k1_MAT_s117
                                 + 0.891613 w_k1_MSFT_s117
                                 + 1.08099 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.04106 w_k1_ADM_s117
                                 + 1.15713 w_k1_AXP_s117
                                 + 0.968808 w_k1_DIS_s117
                                 + 0.959385 w_k1_FDX_s117
                                 + 1.55687 w_k1_IBM_s117
                                 + 1.06232 w_k1_INTC_s117
                                 + 1.24677 w_k1_KO_s117 + 1.68944 w_k1_MAT_s117
                                 + 0.941833 w_k1_MSFT_s117
                                 + 1.16238 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15084 w_k1_ADM_s117
                                 + 0.96676 w_k1_AXP_s117
                                 + 1.50772 w_k1_DIS_s117
                                 + 1.20042 w_k1_FDX_s117 + 1.1249 w_k1_IBM_s117
                                 + 1.05233 w_k1_INTC_s117
                                 + 1.16777 w_k1_KO_s117 + 1.16521 w_k1_MAT_s117
                                 + 1.0344 w_k1_MSFT_s117 + 1.2609 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.3186 w_k1_ADM_s117
                                 + 1.56037 w_k1_AXP_s117
                                 + 1.51135 w_k1_DIS_s117
                                 + 1.49682 w_k1_FDX_s117
                                 + 1.48887 w_k1_IBM_s117
                                 + 1.36362 w_k1_INTC_s117
                                 + 1.26772 w_k1_KO_s117 + 1.54739 w_k1_MAT_s117
                                 + 1.35134 w_k1_MSFT_s117
                                 + 1.27806 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.778077 w_k1_ADM_s117
                                 + 0.793485 w_k1_AXP_s117
                                 + 0.775768 w_k1_DIS_s117
                                 + 0.835829 w_k1_FDX_s117
                                 + 1.00627 w_k1_IBM_s117
                                 + 0.900324 w_k1_INTC_s117
                                 + 0.868401 w_k1_KO_s117
                                 + 0.931032 w_k1_MAT_s117
                                 + 0.827265 w_k1_MSFT_s117
                                 + 0.880206 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.857277 w_k1_ADM_s117
                                 + 0.82057 w_k1_AXP_s117
                                 + 0.75029 w_k1_DIS_s117
                                 + 0.821044 w_k1_FDX_s117
                                 + 1.30266 w_k1_IBM_s117
                                 + 0.718322 w_k1_INTC_s117
                                 + 0.906833 w_k1_KO_s117
                                 + 1.45593 w_k1_MAT_s117
                                 + 0.78444 w_k1_MSFT_s117
                                 + 1.15189 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.13282 w_k1_ADM_s117
                                 + 0.818572 w_k1_AXP_s117
                                 + 0.722005 w_k1_DIS_s117
                                 + 0.922898 w_k1_FDX_s117
                                 + 0.75821 w_k1_IBM_s117
                                 + 0.777153 w_k1_INTC_s117
                                 + 0.702854 w_k1_KO_s117
                                 + 0.747151 w_k1_MAT_s117
                                 + 0.825842 w_k1_MSFT_s117
                                 + 0.982623 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0183292860581825 w_k1_ADM_s000
                                 + 0.0131565872227903 w_k1_ADM_s001
                                 + 0.0153382229151697 w_k1_ADM_s002
                                 + 0.0206505232918232 w_k1_ADM_s003
                                 + 0.0276209601115812 w_k1_ADM_s004
                                 + 0.0363866880878821 w_k1_ADM_s005
                                 + 0.0480831385799551 w_k1_ADM_s006
                                 + 0.0144757604607805 w_k1_ADM_s007
                                 + 0.0753060639047743 w_k1_ADM_s010
                                 + 0.0820216512684534 w_k1_ADM_s011
                                 + 0.0259178971691011 w_k1_ADM_s012
                                 + 0.0225277837053078 w_k1_ADM_s013
                                 + 0.046762009082313 w_k1_ADM_s014
                                 + 0.05713468186735 w_k1_ADM_s015
                                 + 0.0463444421126448 w_k1_ADM_s016
                                 + 0.0635421992058527 w_k1_ADM_s017
                                 + 0.0110696269157091 w_k1_ADM_s100
                                 + 0.0128100250295118 w_k1_ADM_s101
                                 + 0.0223231096686178 w_k1_ADM_s102
                                 + 0.0327363860288232 w_k1_ADM_s103
                                 + 0.0234441657644381 w_k1_ADM_s104
                                 + 0.0159708018547194 w_k1_ADM_s105
                                 + 0.0249920579167023 w_k1_ADM_s106
                                 + 0.0522338702986453 w_k1_ADM_s107
                                 + 0.0196207007096624 w_k1_ADM_s110
                                 + 0.025781979568385 w_k1_ADM_s111
                                 + 0.0225582315148117 w_k1_ADM_s112
                                 + 0.0254775498124576 w_k1_ADM_s113
                                 + 0.0172347464314222 w_k1_ADM_s114
                                 + 0.0339047471790588 w_k1_ADM_s115
                                 + 0.0209021805827784 w_k1_ADM_s116
                                 + 0.0404996606812365 w_k1_ADM_s117
                                 + 0.0165767578306034 w_k1_AXP_s000
                                 + 0.0130787115635588 w_k1_AXP_s001
                                 + 0.0149647347926424 w_k1_AXP_s002
                                 + 0.0228394560201187 w_k1_AXP_s003
                                 + 0.0226397218126747 w_k1_AXP_s004
                                 + 0.0370518021610795 w_k1_AXP_s005
                                 + 0.0445682138230124 w_k1_AXP_s006
                                 + 0.0149973935464169 w_k1_AXP_s007
                                 + 0.0754726111902339 w_k1_AXP_s010
                                 + 0.0887621053719853 w_k1_AXP_s011
                                 + 0.0254175767120753 w_k1_AXP_s012
                                 + 0.0192255299356393 w_k1_AXP_s013
                                 + 0.0524476160834162 w_k1_AXP_s014
                                 + 0.0611053461697123 w_k1_AXP_s015
                                 + 0.0413226055431513 w_k1_AXP_s016
                                 + 0.0709840027035817 w_k1_AXP_s017
                                 + 0.0129751752221879 w_k1_AXP_s100
                                 + 0.0107381231122699 w_k1_AXP_s101
                                 + 0.017398383531857 w_k1_AXP_s102
                                 + 0.0358108462900449 w_k1_AXP_s103
                                 + 0.0271887107108125 w_k1_AXP_s104
                                 + 0.0174392663907589 w_k1_AXP_s105
                                 + 0.0243639094858995 w_k1_AXP_s106
                                 + 0.0545534663580627 w_k1_AXP_s107
                                 + 0.017749500191764 w_k1_AXP_s110
                                 + 0.0279684366117505 w_k1_AXP_s111
                                 + 0.0214143841384555 w_k1_AXP_s112
                                 + 0.0214861417697358 w_k1_AXP_s113
                                 + 0.0164465215428527 w_k1_AXP_s114
                                 + 0.035942667797239 w_k1_AXP_s115
                                 + 0.020200085565581 w_k1_AXP_s116
                                 + 0.0405969718139179 w_k1_AXP_s117
                                 + 0.0184709284235486 w_k1_DIS_s000
                                 + 0.0134761907814972 w_k1_DIS_s001
                                 + 0.0159643873630739 w_k1_DIS_s002
                                 + 0.0223175417305027 w_k1_DIS_s003
                                 + 0.0246848386018349 w_k1_DIS_s004
                                 + 0.0396917015630802 w_k1_DIS_s005
                                 + 0.0471510910851486 w_k1_DIS_s006
                                 + 0.0153866458909398 w_k1_DIS_s007
                                 + 0.079840097244165 w_k1_DIS_s010
                                 + 0.0909655981782748 w_k1_DIS_s011
                                 + 0.0253593565250926 w_k1_DIS_s012
                                 + 0.019305592575427 w_k1_DIS_s013
                                 + 0.0546980555723707 w_k1_DIS_s014
                                 + 0.0643634710873948 w_k1_DIS_s015
                                 + 0.0422268561524398 w_k1_DIS_s016
                                 + 0.0689626579399513 w_k1_DIS_s017
                                 + 0.0112525938982162 w_k1_DIS_s100
                                 + 0.0117342644928605 w_k1_DIS_s101
                                 + 0.0199743230609826 w_k1_DIS_s102
                                 + 0.0337530497136848 w_k1_DIS_s103
                                 + 0.0266288206839049 w_k1_DIS_s104
                                 + 0.0178575237018781 w_k1_DIS_s105
                                 + 0.0265469377066207 w_k1_DIS_s106
                                 + 0.0560303914441199 w_k1_DIS_s107
                                 + 0.0209700575083673 w_k1_DIS_s110
                                 + 0.0275227687317539 w_k1_DIS_s111
                                 + 0.0228150806436872 w_k1_DIS_s112
                                 + 0.0254337290652028 w_k1_DIS_s113
                                 + 0.0184371273734555 w_k1_DIS_s114
                                 + 0.0361623616730696 w_k1_DIS_s115
                                 + 0.0211675838459957 w_k1_DIS_s116
                                 + 0.0405798072045741 w_k1_DIS_s117
                                 + 0.0180623282298279 w_k1_FDX_s000
                                 + 0.0132122006729567 w_k1_FDX_s001
                                 + 0.0157393304746095 w_k1_FDX_s002
                                 + 0.0218684609445276 w_k1_FDX_s003
                                 + 0.0236835388474778 w_k1_FDX_s004
                                 + 0.0389863516212633 w_k1_FDX_s005
                                 + 0.0430304689872559 w_k1_FDX_s006
                                 + 0.0140036429022976 w_k1_FDX_s007
                                 + 0.0791545203717877 w_k1_FDX_s010
                                 + 0.0913753998491011 w_k1_FDX_s011
                                 + 0.0243651235549386 w_k1_FDX_s012
                                 + 0.0191714054096239 w_k1_FDX_s013
                                 + 0.0493331537344112 w_k1_FDX_s014
                                 + 0.0617706406543691 w_k1_FDX_s015
                                 + 0.0425017140166591 w_k1_FDX_s016
                                 + 0.0644400249982527 w_k1_FDX_s017
                                 + 0.0115101990451849 w_k1_FDX_s100
                                 + 0.0121642929523724 w_k1_FDX_s101
                                 + 0.0198277631311628 w_k1_FDX_s102
                                 + 0.0314273135147291 w_k1_FDX_s103
                                 + 0.0249838824348211 w_k1_FDX_s104
                                 + 0.0175818526109314 w_k1_FDX_s105
                                 + 0.0264132242337918 w_k1_FDX_s106
                                 + 0.0539853888728421 w_k1_FDX_s107
                                 + 0.0198903181257107 w_k1_FDX_s110
                                 + 0.0285782477566121 w_k1_FDX_s111
                                 + 0.0214330182238925 w_k1_FDX_s112
                                 + 0.0245998188196685 w_k1_FDX_s113
                                 + 0.01674372709637 w_k1_FDX_s114
                                 + 0.0358427518834934 w_k1_FDX_s115
                                 + 0.0195676425549559 w_k1_FDX_s116
                                 + 0.0423636068183978 w_k1_FDX_s117
                                 + 0.0175522166915426 w_k1_IBM_s000
                                 + 0.0123895686274714 w_k1_IBM_s001
                                 + 0.0164592736652227 w_k1_IBM_s002
                                 + 0.0219152167802934 w_k1_IBM_s003
                                 + 0.0241376925940833 w_k1_IBM_s004
                                 + 0.0418538554823462 w_k1_IBM_s005
                                 + 0.0429510699320595 w_k1_IBM_s006
                                 + 0.0149153109305136 w_k1_IBM_s007
                                 + 0.0772041088187705 w_k1_IBM_s010
                                 + 0.0901228395626253 w_k1_IBM_s011
                                 + 0.0257588787503548 w_k1_IBM_s012
                                 + 0.0196293801539701 w_k1_IBM_s013
                                 + 0.0476002627355677 w_k1_IBM_s014
                                 + 0.0659155802878442 w_k1_IBM_s015
                                 + 0.0440055708671796 w_k1_IBM_s016
                                 + 0.0650685122590761 w_k1_IBM_s017
                                 + 0.0112722979975901 w_k1_IBM_s100
                                 + 0.0122538268708126 w_k1_IBM_s101
                                 + 0.0201367622221674 w_k1_IBM_s102
                                 + 0.0311116372375245 w_k1_IBM_s103
                                 + 0.0273088921356517 w_k1_IBM_s104
                                 + 0.0187091941738105 w_k1_IBM_s105
                                 + 0.0263839714621328 w_k1_IBM_s106
                                 + 0.0524948440651519 w_k1_IBM_s107
                                 + 0.0204670703914764 w_k1_IBM_s110
                                 + 0.0290587275869073 w_k1_IBM_s111
                                 + 0.0232496650202272 w_k1_IBM_s112
                                 + 0.0240570190244691 w_k1_IBM_s113
                                 + 0.0167007140974286 w_k1_IBM_s114
                                 + 0.0353735277347288 w_k1_IBM_s115
                                 + 0.0197986025761707 w_k1_IBM_s116
                                 + 0.043177215288074 w_k1_IBM_s117
                                 + 0.0192325088008249 w_k1_INTC_s000
                                 + 0.0133663820230312 w_k1_INTC_s001
                                 + 0.0192596107777463 w_k1_INTC_s002
                                 + 0.0222571459600462 w_k1_INTC_s003
                                 + 0.026155867109218 w_k1_INTC_s004
                                 + 0.044764046486291 w_k1_INTC_s005
                                 + 0.0425903241148762 w_k1_INTC_s006
                                 + 0.014620651785489 w_k1_INTC_s007
                                 + 0.0810206419777545 w_k1_INTC_s010
                                 + 0.084471235220124 w_k1_INTC_s011
                                 + 0.0254976953307972 w_k1_INTC_s012
                                 + 0.0189575760235794 w_k1_INTC_s013
                                 + 0.0479647143996251 w_k1_INTC_s014
                                 + 0.0672574537932764 w_k1_INTC_s015
                                 + 0.0408376132606481 w_k1_INTC_s016
                                 + 0.0718626365498851 w_k1_INTC_s017
                                 + 0.0114556440412859 w_k1_INTC_s100
                                 + 0.0114094286570944 w_k1_INTC_s101
                                 + 0.0182098515375812 w_k1_INTC_s102
                                 + 0.0296789970717435 w_k1_INTC_s103
                                 + 0.0237059620609887 w_k1_INTC_s104
                                 + 0.0181392545665631 w_k1_INTC_s105
                                 + 0.027245014391393 w_k1_INTC_s106
                                 + 0.0601820539219085 w_k1_INTC_s107
                                 + 0.0197186784774489 w_k1_INTC_s110
                                 + 0.0280524573669897 w_k1_INTC_s111
                                 + 0.0205996980758489 w_k1_INTC_s112
                                 + 0.0261903546012011 w_k1_INTC_s113
                                 + 0.0150201027357256 w_k1_INTC_s114
                                 + 0.0358453868245789 w_k1_INTC_s115
                                 + 0.0175488484564118 w_k1_INTC_s116
                                 + 0.0389229660680264 w_k1_INTC_s117
                                 + 0.0190873067073597 w_k1_KO_s000
                                 + 0.0132492212419602 w_k1_KO_s001
                                 + 0.0167320727308708 w_k1_KO_s002
                                 + 0.0254061742228096 w_k1_KO_s003
                                 + 0.0273991101368272 w_k1_KO_s004
                                 + 0.0419721267659322 w_k1_KO_s005
                                 + 0.0474967176441006 w_k1_KO_s006
                                 + 0.0147131954729045 w_k1_KO_s007
                                 + 0.0751542287630943 w_k1_KO_s010
                                 + 0.0871182187762678 w_k1_KO_s011
                                 + 0.025975510864455 w_k1_KO_s012
                                 + 0.0196616501932337 w_k1_KO_s013
                                 + 0.0463790024713387 w_k1_KO_s014
                                 + 0.0631848765293818 w_k1_KO_s015
                                 + 0.0425287025007807 w_k1_KO_s016
                                 + 0.058814147916384 w_k1_KO_s017
                                 + 0.0106611608162161 w_k1_KO_s100
                                 + 0.0120900114158722 w_k1_KO_s101
                                 + 0.0210827266776618 w_k1_KO_s102
                                 + 0.0318677192855171 w_k1_KO_s103
                                 + 0.0257678496604614 w_k1_KO_s104
                                 + 0.0186367016183814 w_k1_KO_s105
                                 + 0.0264899916166534 w_k1_KO_s106
                                 + 0.0499368714635986 w_k1_KO_s107
                                 + 0.0214446639621748 w_k1_KO_s110
                                 + 0.0290634944486552 w_k1_KO_s111
                                 + 0.0241517778095566 w_k1_KO_s112
                                 + 0.0271736605082119 w_k1_KO_s113
                                 + 0.0183224155664416 w_k1_KO_s114
                                 + 0.0360937341650953 w_k1_KO_s115
                                 + 0.0184078300695415 w_k1_KO_s116
                                 + 0.0393201639985771 w_k1_KO_s117
                                 + 0.0172227800322443 w_k1_MAT_s000
                                 + 0.0133065265026962 w_k1_MAT_s001
                                 + 0.0132169959423425 w_k1_MAT_s002
                                 + 0.0207200918482914 w_k1_MAT_s003
                                 + 0.0236672142216165 w_k1_MAT_s004
                                 + 0.0454872766133774 w_k1_MAT_s005
                                 + 0.0534884259104479 w_k1_MAT_s006
                                 + 0.0184391711867049 w_k1_MAT_s007
                                 + 0.0691621458226447 w_k1_MAT_s010
                                 + 0.106584029593645 w_k1_MAT_s011
                                 + 0.0208828583535498 w_k1_MAT_s012
                                 + 0.0183832294864453 w_k1_MAT_s013
                                 + 0.0469962988492859 w_k1_MAT_s014
                                 + 0.0609394349988504 w_k1_MAT_s015
                                 + 0.048003381069897 w_k1_MAT_s016
                                 + 0.0587903505895572 w_k1_MAT_s017
                                 + 0.0109068744851946 w_k1_MAT_s100
                                 + 0.00873185211784482 w_k1_MAT_s101
                                 + 0.0209424619770226 w_k1_MAT_s102
                                 + 0.0271838227547492 w_k1_MAT_s103
                                 + 0.0285467449421306 w_k1_MAT_s104
                                 + 0.0160709112518091 w_k1_MAT_s105
                                 + 0.0264448105565667 w_k1_MAT_s106
                                 + 0.0512375308615786 w_k1_MAT_s107
                                 + 0.0212522723921571 w_k1_MAT_s110
                                 + 0.0277737869008667 w_k1_MAT_s111
                                 + 0.0215013495894931 w_k1_MAT_s112
                                 + 0.0242986794943527 w_k1_MAT_s113
                                 + 0.0167046701289254 w_k1_MAT_s114
                                 + 0.0338853994657631 w_k1_MAT_s115
                                 + 0.0206779364832699 w_k1_MAT_s116
                                 + 0.0447688688476894 w_k1_MAT_s117
                                 + 0.0175068031531101 w_k1_MSFT_s000
                                 + 0.0130766620003303 w_k1_MSFT_s001
                                 + 0.0174436009457207 w_k1_MSFT_s002
                                 + 0.0212545708351575 w_k1_MSFT_s003
                                 + 0.0230581451092449 w_k1_MSFT_s004
                                 + 0.0418218218512631 w_k1_MSFT_s005
                                 + 0.0423426189932718 w_k1_MSFT_s006
                                 + 0.0146470966638322 w_k1_MSFT_s007
                                 + 0.0841343887814228 w_k1_MSFT_s010
                                 + 0.0886345563942217 w_k1_MSFT_s011
                                 + 0.0251313655706625 w_k1_MSFT_s012
                                 + 0.0192908807839321 w_k1_MSFT_s013
                                 + 0.0477789843914496 w_k1_MSFT_s014
                                 + 0.0651652363443445 w_k1_MSFT_s015
                                 + 0.0431929835550509 w_k1_MSFT_s016
                                 + 0.0671286186387338 w_k1_MSFT_s017
                                 + 0.0106202531770521 w_k1_MSFT_s100
                                 + 0.0120066607145778 w_k1_MSFT_s101
                                 + 0.0205434057654273 w_k1_MSFT_s102
                                 + 0.0320983119657887 w_k1_MSFT_s103
                                 + 0.0248392048852859 w_k1_MSFT_s104
                                 + 0.0188311128532839 w_k1_MSFT_s105
                                 + 0.0254359667042958 w_k1_MSFT_s106
                                 + 0.0537348312572694 w_k1_MSFT_s107
                                 + 0.0205645248112461 w_k1_MSFT_s110
                                 + 0.0278351568893243 w_k1_MSFT_s111
                                 + 0.021977330021696 w_k1_MSFT_s112
                                 + 0.0240430652072177 w_k1_MSFT_s113
                                 + 0.0159149214841795 w_k1_MSFT_s114
                                 + 0.036093576632359 w_k1_MSFT_s115
                                 + 0.0203126049171169 w_k1_MSFT_s116
                                 + 0.0385521754467696 w_k1_MSFT_s117
                                 + 0.0184457805528184 w_k1_PG_s000
                                 + 0.013062715087652 w_k1_PG_s001
                                 + 0.0157301418833372 w_k1_PG_s002
                                 + 0.0229261025069925 w_k1_PG_s003
                                 + 0.0234265696750309 w_k1_PG_s004
                                 + 0.0393885005334893 w_k1_PG_s005
                                 + 0.0434555233485271 w_k1_PG_s006
                                 + 0.0141637663673938 w_k1_PG_s007
                                 + 0.0751417251766368 w_k1_PG_s010
                                 + 0.0818108227938804 w_k1_PG_s011
                                 + 0.0232028706703411 w_k1_PG_s012
                                 + 0.0173290072988955 w_k1_PG_s013
                                 + 0.0491349688265023 w_k1_PG_s014
                                 + 0.0577817223002675 w_k1_PG_s015
                                 + 0.0422694211478741 w_k1_PG_s016
                                 + 0.0605871979330712 w_k1_PG_s017
                                 + 0.0106747770459083 w_k1_PG_s100
                                 + 0.0115151385679721 w_k1_PG_s101
                                 + 0.0200539651306112 w_k1_PG_s102
                                 + 0.0312379518840066 w_k1_PG_s103
                                 + 0.0232454868188209 w_k1_PG_s104
                                 + 0.017166552292559 w_k1_PG_s105
                                 + 0.0261219086409682 w_k1_PG_s106
                                 + 0.049776331405188 w_k1_PG_s107
                                 + 0.0208269863886058 w_k1_PG_s110
                                 + 0.0269124562904311 w_k1_PG_s111
                                 + 0.0192596398309204 w_k1_PG_s112
                                 + 0.0229057829767788 w_k1_PG_s113
                                 + 0.0160661597881521 w_k1_PG_s114
                                 + 0.0347243458207508 w_k1_PG_s115
                                 + 0.0187824984934946 w_k1_PG_s116
                                 + 0.0434292783771815 w_k1_PG_s117 >= 1.16
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
