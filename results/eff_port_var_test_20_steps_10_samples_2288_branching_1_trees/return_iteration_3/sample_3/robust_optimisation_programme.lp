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
 w_k1_ADM_s0_return_constraint:  1.21563 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.976366 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.820874 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06362 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.08623 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.876619 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.823515 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.10982 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.06734 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.906247 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.17179 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.24484 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.39048 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.777885 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.769749 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.26147 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.736928 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.952105 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.972412 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.11031 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.17893 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.28023 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.715799 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.32181 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.835153 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.901599 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.26803 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.84026 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.43674 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.44615 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.67769 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.86697 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10907 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.15577 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.990455 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.911507 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.47033 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.32723 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29074 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.918268 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.737684 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.11639 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.2947 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.871262 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.766047 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.34366 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.14804 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.982583 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.840417 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.57448 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.577 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.855325 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.626226 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.23654 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.890888 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.08375 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.011 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.00463 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.68774 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.3788 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.745636 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.39435 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.10547 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.07664 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.0186 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.758792 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.37707 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.38949 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.744217 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.846759 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.26383 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.29976 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.943744 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.888878 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.6562 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.34402 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.42832 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.932201 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.681808 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.13573 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.36203 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.88827 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.627598 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.3101 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.15862 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.09226 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.903055 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.37164 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.8429 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.69204 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.598829 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.34044 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.75091 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.923276 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.920978 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.12516 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.73853 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.53305 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.7141 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.27572 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.937654 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.876109 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.985422 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.738965 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.54556 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.61238 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.577106 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.798724 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.13526 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.03859 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.886552 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.782461 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.62173 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.42295 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.25149 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.947132 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.792929 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.03495 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.16132 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.951413 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.765257 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.26921 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.997356 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.10082 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.09715 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.30309 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.4677 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.795695 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.716227 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.28947 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.772844 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.97611 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.1186 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.08393 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.64105 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.27794 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.730172 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.23138 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.935356 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.01442 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.11961 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.81357 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.43701 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.32404 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.669534 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.749215 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.18758 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.959925 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.893206 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.893502 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.49547 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.18641 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.26726 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.947371 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.723043 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.101 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.36284 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.889052 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.667405 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.807927 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.29911 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.915335 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.586383 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.28306 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.40645 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.074 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.797283 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.10953 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.28158 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.2851 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.729577 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.09458 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.03467 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.80201 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.18109 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.09323 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.5603 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.992182 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.728882 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.795051 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.911596 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.64284 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.881588 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.5585 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.803646 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.44716 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.07256 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.617406 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.897142 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.33851 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1837 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.963128 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.720619 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.926344 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.00526 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.96402 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.8465 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.25281 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.30981 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.30683 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.13517 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.21289 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.56485 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.99206 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.923536 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.428 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.03459 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.26257 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.0524 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.17197 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.55846 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.64194 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.900591 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.29639 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.24413 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.18892 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.34413 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.874648 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.45811 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.39575 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.831493 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.967293 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.09587 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.933808 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.95668 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.893285 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.33583 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.17604 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2947 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.935975 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.687372 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.09617 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.24734 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.900991 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.764546 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.09072 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.21876 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.990141 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.745459 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.29365 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.61842 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.942342 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.76769 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.13336 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.03092 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.03537 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.741198 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.19123 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.21149 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.55909 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.95818 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.2075 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.13392 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.947615 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.880831 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.790184 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.11437 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.70728 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.80946 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.11763 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.93818 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.26914 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.10802 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.717561 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.27714 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.38122 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.29291 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.896746 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.635504 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.24986 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.33653 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.837389 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.568877 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.814671 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.36608 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.0333 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.583612 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.79602 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.74709 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.37941 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.655171 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.992632 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23202 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.11921 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.551328 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.0644 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.2873 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.51418 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.13091 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.49091 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.77825 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.25153 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.732102 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.784139 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.913859 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.80056 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.871887 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.60797 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.996704 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.55438 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.99159 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.636513 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.18576 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.33093 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.17831 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.96214 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.834164 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.07793 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.17588 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.940477 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.708928 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.19353 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.990964 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.09927 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.08659 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.1505 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.42471 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.761171 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.79144 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.21321 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.825767 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.03717 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.03222 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.14537 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.42099 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.33604 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.818052 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.12247 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.898706 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.936501 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.05429 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.83235 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.28761 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.3424 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.753972 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.856427 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.99636 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.951045 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.990486 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.899795 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.40428 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.22865 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15541 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.949076 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.855367 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10599 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.1996 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.940989 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.807185 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08266 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.01059 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.00769 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.922048 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26306 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.27425 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.884074 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.843741 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.13277 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.831231 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.963411 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.924247 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.06078 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.23397 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.19966 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.898612 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.09371 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.969059 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.976727 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.916961 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.813563 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09458 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.21076 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.823237 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.07449 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.11323 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.17679 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.933027 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.849525 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.16137 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.22583 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0349999960014046 z_k1_s0000
                                 - 0.0800000997175039 z_k1_s0001
                                 - 0.114000161147444 z_k1_s0002
                                 - 0.0290000075725931 z_k1_s0003
                                 - 0.036000006772874 z_k1_s0004
                                 - 0.0290000075725931 z_k1_s0005
                                 - 0.0309999910155294 z_k1_s0006
                                 - 0.0269999860296542 z_k1_s0007
                                 - 0.0859999371087083 z_k1_s0010
                                 - 0.0659999376904236 z_k1_s0011
                                 - 0.130000029768804 z_k1_s0012
                                 - 0.0599999311549476 z_k1_s0013
                                 - 0.0839996888972259 z_k1_s0014
                                 - 0.0339999252011868 z_k1_s0015
                                 - 0.0749995784441521 z_k1_s0016
                                 - 0.136000103404186 z_k1_s0017
                                 - 0.121999846397818 z_k1_s0020
                                 - 0.14600002129533 z_k1_s0021
                                 - 0.0399998989633227 z_k1_s0022
                                 - 0.0900001038667985 z_k1_s0023
                                 - 0.100000115407554 z_k1_s0024
                                 - 0.0619998507531352 z_k1_s0025
                                 - 0.124999776260195 z_k1_s0026
                                 - 0.0509999116581537 z_k1_s0027
                                 - 0.15099965036589 z_k1_s0030
                                 - 0.115000302026597 z_k1_s0031
                                 - 0.0559999308826711 z_k1_s0032
                                 - 0.0810003382621226 z_k1_s0033
                                 - 0.100999798006301 z_k1_s0034
                                 - 0.044999947323573 z_k1_s0035
                                 - 0.13299957519653 z_k1_s0036
                                 - 0.119999966462785 z_k1_s0037
                                 - 0.0319999887358752 z_k1_s0040
                                 - 0.0790001619416726 z_k1_s0041
                                 - 0.0290000070418852 z_k1_s0042
                                 - 0.0439999155118355 z_k1_s0043
                                 - 0.0470000352058115 z_k1_s0044
                                 - 0.0599999788797661 z_k1_s0045
                                 - 0.0380000211238483 z_k1_s0046
                                 - 0.0159999943679376 z_k1_s0047
                                 - 0.0459999360603341 z_k1_s0050
                                 - 0.0340000246881388 z_k1_s0051
                                 - 0.0660001823476866 z_k1_s0052
                                 - 0.0240000197445707 z_k1_s0053
                                 - 0.0640002365190234 z_k1_s0054
                                 - 0.0340000246881388 z_k1_s0055
                                 - 0.0360000493168741 z_k1_s0056
                                 - 0.0899999262920051 z_k1_s0057
                                 - 0.0259998873479092 z_k1_s0060
                                 - 0.0319998967358846 z_k1_s0061
                                 - 0.0319998967358846 z_k1_s0062
                                 - 0.0209999868579187 z_k1_s0063
                                 - 0.0619999436757618 z_k1_s0064
                                 - 0.0169999882659343 z_k1_s0065
                                 - 0.0259998873479092 z_k1_s0066
                                 - 0.0140000065719442 z_k1_s0067
                                 - 0.119999981534533 z_k1_s0070
                                 - 0.0429999553832288 z_k1_s0071
                                 - 0.124000011319001 z_k1_s0072
                                 - 0.101999847504429 z_k1_s0073
                                 - 0.103000310950293 z_k1_s0074
                                 - 0.197999650332157 z_k1_s0075
                                 - 0.156000249594742 z_k1_s0076
                                 - 0.0659999442440186 z_k1_s0077
                                 - 0.394002535730383 z_k1_s0100
                                 - 0.373001783403114 z_k1_s0101
                                 - 0.272000796740426 z_k1_s0102
                                 - 0.649002197109327 z_k1_s0103
                                 - 0.616001014880762 z_k1_s0104
                                 - 0.356002227045567 z_k1_s0105
                                 - 0.189000454929682 z_k1_s0106
                                 - 0.309000648532418 z_k1_s0107
                                 - 0.272999894276457 z_k1_s0110
                                 - 0.132999854878267 z_k1_s0111
                                 - 0.290000168539115 z_k1_s0112
                                 - 0.190000488111862 z_k1_s0113
                                 - 0.0690000107248217 z_k1_s0114
                                 - 0.101000054501553 z_k1_s0115
                                 - 0.0899999557845457 z_k1_s0116
                                 - 0.0710000286733686 z_k1_s0117
                                 - 0.336999291065302 z_k1_s0120
                                 - 0.704998360579144 z_k1_s0121
                                 - 0.240999495622658 z_k1_s0122
                                 - 0.157999645038982 z_k1_s0123
                                 - 0.299999177821228 z_k1_s0124
                                 - 0.151999602942706 z_k1_s0125
                                 - 0.499997654037852 z_k1_s0126
                                 - 0.533997599884159 z_k1_s0127
                                 - 0.318000556732613 z_k1_s0130
                                 - 0.0960000543532542 z_k1_s0131
                                 - 0.306999273952694 z_k1_s0132
                                 - 0.303999837285342 z_k1_s0133
                                 - 0.284000244503368 z_k1_s0134
                                 - 0.377999335078536 z_k1_s0135
                                 - 0.146999919997189 z_k1_s0136
                                 - 0.174999952591886 z_k1_s0137
                                 - 0.392999240022278 z_k1_s0140
                                 - 0.630997531898977 z_k1_s0141
                                 - 0.619996807995711 z_k1_s0142
                                 - 0.900996705105706 z_k1_s0143
                                 - 0.815999275110269 z_k1_s0144
                                 - 0.418999363660314 z_k1_s0145
                                 - 0.391999446339641 z_k1_s0146
                                 - 0.816997073196733 z_k1_s0147
                                 - 0.138000041961761 z_k1_s0150
                                 - 0.138999413396623 z_k1_s0151
                                 - 0.166000070127905 z_k1_s0152
                                 - 0.293000246310028 z_k1_s0153
                                 - 0.21399973098437 z_k1_s0154
                                 - 0.140999512265578 z_k1_s0155
                                 - 0.0739999969534228 z_k1_s0156
                                 - 0.190999271990999 z_k1_s0157
                                 - 0.276999782397958 z_k1_s0160
                                 - 0.143999392592992 z_k1_s0161
                                 - 0.182000096680329 z_k1_s0162
                                 - 0.211000233667896 z_k1_s0163
                                 - 0.127000103938484 z_k1_s0164
                                 - 0.171999620236193 z_k1_s0165
                                 - 0.103999976217304 z_k1_s0166
                                 - 0.166000164169997 z_k1_s0167
                                 - 0.278000562558096 z_k1_s0170
                                 - 0.287999940490688 z_k1_s0171
                                 - 0.0940001045770442 z_k1_s0172
                                 - 0.198000380688104 z_k1_s0173
                                 - 0.193000294921096 z_k1_s0174
                                 - 0.329001358021434 z_k1_s0175
                                 - 0.364001164789066 z_k1_s0176
                                 - 0.240000942410686 z_k1_s0177
                                 - 0.826996887329642 z_k1_s1000
                                 - 0.530998315712422 z_k1_s1001
                                 - 0.581999558584071 z_k1_s1002
                                 - 0.604997016869868 z_k1_s1003
                                 - 0.172999274614429 z_k1_s1004
                                 - 0.305998793158794 z_k1_s1005
                                 - 0.233999232232678 z_k1_s1006
                                 - 0.508999109096687 z_k1_s1007
                                 - 0.379001376203826 z_k1_s1010
                                 - 0.127999909019414 z_k1_s1011
                                 - 0.527999860692619 z_k1_s1012
                                 - 0.260000009042589 z_k1_s1013
                                 - 0.209000001045004 z_k1_s1014
                                 - 0.446000260316171 z_k1_s1015
                                 - 0.258999961291649 z_k1_s1016
                                 - 0.488000647655417 z_k1_s1017
                                 - 0.201000559334745 z_k1_s1020
                                 - 0.218000374141133 z_k1_s1021
                                 - 0.241000655645108 z_k1_s1022
                                 - 0.51300040605351 z_k1_s1023
                                 - 0.743000206086376 z_k1_s1024
                                 - 0.165000472655419 z_k1_s1025
                                 - 0.332000497875323 z_k1_s1026
                                 - 0.602000394218552 z_k1_s1027
                                 - 0.268998597930955 z_k1_s1030
                                 - 0.102999772876457 z_k1_s1031
                                 - 0.480000250393231 z_k1_s1032
                                 - 0.22899986622964 z_k1_s1033
                                 - 0.128999721981184 z_k1_s1034
                                 - 0.427000107641646 z_k1_s1035
                                 - 0.390000341569299 z_k1_s1036
                                 - 0.182999667275543 z_k1_s1037
                                 - 0.173999288695573 z_k1_s1040
                                 - 0.493996447360515 z_k1_s1041
                                 - 0.354999189246429 z_k1_s1042
                                 - 0.445997542212474 z_k1_s1043
                                 - 0.60999551595982 z_k1_s1044
                                 - 0.491997350225117 z_k1_s1045
                                 - 0.147999367392923 z_k1_s1046
                                 - 0.709994949509778 z_k1_s1047
                                 - 0.369999887226039 z_k1_s1050
                                 - 0.848997096050156 z_k1_s1051
                                 - 0.471999858972135 z_k1_s1052
                                 - 0.374999942404732 z_k1_s1053
                                 - 0.78100096121687 z_k1_s1054
                                 - 0.507999417059279 z_k1_s1055
                                 - 0.756999857159703 z_k1_s1056
                                 - 1.13299792240143 z_k1_s1057
                                 - 0.460999238272165 z_k1_s1060
                                 - 0.204999218003528 z_k1_s1061
                                 - 0.293998404494795 z_k1_s1062
                                 - 0.222999114486906 z_k1_s1063
                                 - 0.317999790733091 z_k1_s1064
                                 - 0.38099842812899 z_k1_s1065
                                 - 0.648997712522596 z_k1_s1066
                                 - 0.326999596075362 z_k1_s1067
                                 - 0.2689994045841 z_k1_s1070
                                 - 0.225000513074208 z_k1_s1071
                                 - 0.126999748589469 z_k1_s1072
                                 - 0.365999338363245 z_k1_s1073
                                 - 0.256999706899584 z_k1_s1074
                                 - 0.159999773120439 z_k1_s1075
                                 - 0.344999391916148 z_k1_s1076
                                 - 0.152999664171621 z_k1_s1077
                                 - 1.66400442098503 z_k1_s1100
                                 - 0.40100161815664 z_k1_s1101
                                 - 1.40600871944859 z_k1_s1102
                                 - 1.16100578799732 z_k1_s1103
                                 - 0.692001971439409 z_k1_s1104
                                 - 1.07999992937477 z_k1_s1105
                                 - 1.01800154678268 z_k1_s1106
                                 - 0.955006297884891 z_k1_s1107
                                 - 0.323000383831148 z_k1_s1110
                                 - 0.235000308038483 z_k1_s1111
                                 - 0.258000292635391 z_k1_s1112
                                 - 0.600001925269689 z_k1_s1113
                                 - 0.345001177455147 z_k1_s1114
                                 - 0.146000196171999 z_k1_s1115
                                 - 0.230000127669392 z_k1_s1116
                                 - 0.680001994172112 z_k1_s1117
                                 - 0.377001085835219 z_k1_s1120
                                 - 0.933001486364375 z_k1_s1121
                                 - 0.321000786740235 z_k1_s1122
                                 - 0.755999185119523 z_k1_s1123
                                 - 1.51200407925405 z_k1_s1124
                                 - 0.631000301563982 z_k1_s1125
                                 - 0.316000831398014 z_k1_s1126
                                 - 0.863003253376273 z_k1_s1127
                                 - 0.425001613679481 z_k1_s1130
                                 - 0.242000395651117 z_k1_s1131
                                 - 0.612999250918328 z_k1_s1132
                                 - 0.504001672158498 z_k1_s1133
                                 - 0.324000206158216 z_k1_s1134
                                 - 0.385999515909408 z_k1_s1135
                                 - 0.666000444886352 z_k1_s1136
                                 - 0.641001244317814 z_k1_s1137
                                 - 1.38799636095357 z_k1_s1140
                                 - 0.854995455257749 z_k1_s1141
                                 - 0.465999392408053 z_k1_s1142
                                 - 1.06600181139165 z_k1_s1143
                                 - 0.511999161344708 z_k1_s1144
                                 - 1.73299654447582 z_k1_s1145
                                 - 0.533999017505764 z_k1_s1146
                                 - 1.11199774733364 z_k1_s1147
                                 - 1.84399596792333 z_k1_s1150
                                 - 1.12299721442098 z_k1_s1151
                                 - 0.735999466783901 z_k1_s1152
                                 - 1.78000226852946 z_k1_s1153
                                 - 0.671999312694124 z_k1_s1154
                                 - 1.09199623209213 z_k1_s1155
                                 - 1.10999561267093 z_k1_s1156
                                 - 0.863999774963455 z_k1_s1157
                                 - 1.12300274765462 z_k1_s1160
                                 - 0.346000337956889 z_k1_s1161
                                 - 0.889998741665031 z_k1_s1162
                                 - 1.04299932397431 z_k1_s1163
                                 - 0.398000532547624 z_k1_s1164
                                 - 0.526998056951351 z_k1_s1165
                                 - 0.333000162384114 z_k1_s1166
                                 - 0.417000203345873 z_k1_s1167
                                 - 1.25900742812235 z_k1_s1170
                                 - 1.74700527490571 z_k1_s1171
                                 - 0.553001478171656 z_k1_s1172
                                 - 1.27899969636989 z_k1_s1173
                                 - 1.14400395305934 z_k1_s1174
                                 - 1.07700283785137 z_k1_s1175
                                 - 1.29000677343905 z_k1_s1176
                                 - 0.366001435346406 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.810199 w_k1_ADM_s000
                                 + 0.848293 w_k1_AXP_s000
                                 + 0.757983 w_k1_DIS_s000
                                 + 0.859886 w_k1_FDX_s000
                                 + 1.50226 w_k1_IBM_s000
                                 + 1.21076 w_k1_INTC_s000
                                 + 0.973688 w_k1_KO_s000
                                 + 0.855491 w_k1_MAT_s000
                                 + 0.877473 w_k1_MSFT_s000
                                 + 0.933779 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.851227 w_k1_ADM_s000
                                 + 0.566234 w_k1_AXP_s000
                                 + 0.650542 w_k1_DIS_s000
                                 + 0.835775 w_k1_FDX_s000
                                 + 0.776644 w_k1_IBM_s000
                                 + 1.04276 w_k1_INTC_s000
                                 + 0.759176 w_k1_KO_s000
                                 + 0.741971 w_k1_MAT_s000
                                 + 1.03832 w_k1_MSFT_s000
                                 + 0.861881 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.12193 w_k1_ADM_s000
                                 + 1.04814 w_k1_AXP_s000
                                 + 0.972608 w_k1_DIS_s000
                                 + 1.09865 w_k1_FDX_s000
                                 + 1.06296 w_k1_IBM_s000
                                 + 1.27611 w_k1_INTC_s000
                                 + 0.976849 w_k1_KO_s000
                                 + 0.902127 w_k1_MAT_s000
                                 + 1.19286 w_k1_MSFT_s000
                                 + 0.90321 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.13442 w_k1_ADM_s000
                                 + 1.0927 w_k1_AXP_s000 + 1.55494 w_k1_DIS_s000
                                 + 1.22599 w_k1_FDX_s000
                                 + 1.01261 w_k1_IBM_s000
                                 + 1.54378 w_k1_INTC_s000
                                 + 1.38821 w_k1_KO_s000 + 1.61691 w_k1_MAT_s000
                                 + 1.52062 w_k1_MSFT_s000
                                 + 1.07027 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.1008 w_k1_ADM_s000
                                 + 0.67601 w_k1_AXP_s000
                                 + 0.870254 w_k1_DIS_s000
                                 + 1.07071 w_k1_FDX_s000
                                 + 0.678784 w_k1_IBM_s000
                                 + 1.11145 w_k1_INTC_s000
                                 + 0.932841 w_k1_KO_s000
                                 + 0.932187 w_k1_MAT_s000
                                 + 1.32939 w_k1_MSFT_s000
                                 + 1.01847 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.972773 w_k1_ADM_s000
                                 + 1.29868 w_k1_AXP_s000
                                 + 1.22853 w_k1_DIS_s000
                                 + 1.08684 w_k1_FDX_s000
                                 + 1.53192 w_k1_IBM_s000
                                 + 1.16993 w_k1_INTC_s000
                                 + 1.31272 w_k1_KO_s000 + 2.13968 w_k1_MAT_s000
                                 + 1.18544 w_k1_MSFT_s000
                                 + 1.17789 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.923956 w_k1_ADM_s000
                                 + 0.768403 w_k1_AXP_s000
                                 + 0.688816 w_k1_DIS_s000
                                 + 0.797616 w_k1_FDX_s000
                                 + 1.16054 w_k1_IBM_s000
                                 + 0.89334 w_k1_INTC_s000
                                 + 0.866033 w_k1_KO_s000 + 1.1832 w_k1_MAT_s000
                                 + 1.02794 w_k1_MSFT_s000
                                 + 0.862778 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.43134 w_k1_ADM_s000
                                 + 1.38796 w_k1_AXP_s000 + 1.817 w_k1_DIS_s000
                                 + 1.59675 w_k1_FDX_s000
                                 + 1.13651 w_k1_IBM_s000
                                 + 1.5873 w_k1_INTC_s000 + 1.22214 w_k1_KO_s000
                                 + 1.28626 w_k1_MAT_s000
                                 + 1.46806 w_k1_MSFT_s000
                                 + 1.36426 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.07208 w_k1_ADM_s001
                                 + 1.12398 w_k1_AXP_s001
                                 + 0.987014 w_k1_DIS_s001
                                 + 1.15862 w_k1_FDX_s001
                                 + 0.810326 w_k1_IBM_s001
                                 + 1.13138 w_k1_INTC_s001
                                 + 0.914261 w_k1_KO_s001
                                 + 1.36722 w_k1_MAT_s001
                                 + 1.13261 w_k1_MSFT_s001
                                 + 1.08165 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.968014 w_k1_ADM_s001
                                 + 0.799213 w_k1_AXP_s001
                                 + 0.604322 w_k1_DIS_s001
                                 + 0.771567 w_k1_FDX_s001
                                 + 1.00167 w_k1_IBM_s001
                                 + 0.832388 w_k1_INTC_s001
                                 + 0.826751 w_k1_KO_s001
                                 + 1.11038 w_k1_MAT_s001
                                 + 0.708103 w_k1_MSFT_s001
                                 + 0.842685 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.03876 w_k1_ADM_s001
                                 + 1.3053 w_k1_AXP_s001 + 1.57737 w_k1_DIS_s001
                                 + 1.33224 w_k1_FDX_s001
                                 + 0.988437 w_k1_IBM_s001
                                 + 1.34149 w_k1_INTC_s001
                                 + 1.19508 w_k1_KO_s001 + 1.41938 w_k1_MAT_s001
                                 + 1.33472 w_k1_MSFT_s001
                                 + 1.18811 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.03311 w_k1_ADM_s001
                                 + 1.23149 w_k1_AXP_s001
                                 + 1.02889 w_k1_DIS_s001
                                 + 0.937923 w_k1_FDX_s001
                                 + 1.45844 w_k1_IBM_s001
                                 + 1.1051 w_k1_INTC_s001 + 1.0039 w_k1_KO_s001
                                 + 1.65893 w_k1_MAT_s001
                                 + 0.897142 w_k1_MSFT_s001 + 0.952 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0853 w_k1_ADM_s001
                                 + 1.01579 w_k1_AXP_s001
                                 + 0.809117 w_k1_DIS_s001
                                 + 0.985549 w_k1_FDX_s001
                                 + 0.973315 w_k1_IBM_s001
                                 + 0.976477 w_k1_INTC_s001
                                 + 0.874966 w_k1_KO_s001
                                 + 1.10033 w_k1_MAT_s001
                                 + 0.910208 w_k1_MSFT_s001
                                 + 1.05171 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.37074 w_k1_ADM_s001
                                 + 1.30566 w_k1_AXP_s001 + 1.1032 w_k1_DIS_s001
                                 + 1.22262 w_k1_FDX_s001
                                 + 1.56239 w_k1_IBM_s001
                                 + 1.38952 w_k1_INTC_s001
                                 + 1.20882 w_k1_KO_s001 + 1.66544 w_k1_MAT_s001
                                 + 1.14706 w_k1_MSFT_s001 + 1.1644 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.12916 w_k1_ADM_s001
                                 + 0.843204 w_k1_AXP_s001
                                 + 1.05071 w_k1_DIS_s001
                                 + 1.03749 w_k1_FDX_s001
                                 + 0.821063 w_k1_IBM_s001
                                 + 1.09461 w_k1_INTC_s001
                                 + 1.04518 w_k1_KO_s001
                                 + 0.944259 w_k1_MAT_s001
                                 + 1.01319 w_k1_MSFT_s001
                                 + 0.938686 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.792639 w_k1_ADM_s001
                                 + 0.704382 w_k1_AXP_s001
                                 + 0.634056 w_k1_DIS_s001
                                 + 0.850971 w_k1_FDX_s001
                                 + 0.867409 w_k1_IBM_s001
                                 + 0.885074 w_k1_INTC_s001
                                 + 0.791074 w_k1_KO_s001
                                 + 1.05013 w_k1_MAT_s001
                                 + 0.923377 w_k1_MSFT_s001
                                 + 0.84845 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.07981 w_k1_ADM_s002
                                 + 1.11675 w_k1_AXP_s002
                                 + 1.30362 w_k1_DIS_s002
                                 + 1.18344 w_k1_FDX_s002
                                 + 1.01009 w_k1_IBM_s002
                                 + 1.04477 w_k1_INTC_s002 + 1.2076 w_k1_KO_s002
                                 + 0.87195 w_k1_MAT_s002
                                 + 1.24161 w_k1_MSFT_s002
                                 + 1.08824 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.840881 w_k1_ADM_s002
                                 + 0.790513 w_k1_AXP_s002
                                 + 0.729648 w_k1_DIS_s002
                                 + 0.732055 w_k1_FDX_s002
                                 + 1.10051 w_k1_IBM_s002
                                 + 0.738851 w_k1_INTC_s002
                                 + 0.91519 w_k1_KO_s002 + 1.21383 w_k1_MAT_s002
                                 + 0.743252 w_k1_MSFT_s002
                                 + 0.918769 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.12206 w_k1_ADM_s002
                                 + 1.51119 w_k1_AXP_s002
                                 + 1.00498 w_k1_DIS_s002
                                 + 1.26607 w_k1_FDX_s002
                                 + 1.08468 w_k1_IBM_s002
                                 + 1.09061 w_k1_INTC_s002
                                 + 1.02718 w_k1_KO_s002 + 1.6768 w_k1_MAT_s002
                                 + 1.06966 w_k1_MSFT_s002
                                 + 1.00947 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.00738 w_k1_ADM_s002
                                 + 1.12999 w_k1_AXP_s002
                                 + 0.97813 w_k1_DIS_s002
                                 + 0.938778 w_k1_FDX_s002
                                 + 1.70327 w_k1_IBM_s002
                                 + 1.03886 w_k1_INTC_s002
                                 + 1.40594 w_k1_KO_s002 + 1.84793 w_k1_MAT_s002
                                 + 0.965041 w_k1_MSFT_s002
                                 + 1.1642 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.771496 w_k1_ADM_s002
                                 + 0.93598 w_k1_AXP_s002
                                 + 0.789439 w_k1_DIS_s002
                                 + 0.926494 w_k1_FDX_s002
                                 + 0.806515 w_k1_IBM_s002
                                 + 0.946605 w_k1_INTC_s002
                                 + 0.724864 w_k1_KO_s002
                                 + 0.802161 w_k1_MAT_s002
                                 + 0.904095 w_k1_MSFT_s002
                                 + 0.897716 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.712811 w_k1_ADM_s002
                                 + 0.803346 w_k1_AXP_s002
                                 + 0.762311 w_k1_DIS_s002
                                 + 1.01803 w_k1_FDX_s002
                                 + 0.680562 w_k1_IBM_s002
                                 + 1.00972 w_k1_INTC_s002
                                 + 0.825914 w_k1_KO_s002
                                 + 0.867383 w_k1_MAT_s002
                                 + 1.16905 w_k1_MSFT_s002
                                 + 0.936706 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.786377 w_k1_ADM_s002
                                 + 0.893217 w_k1_AXP_s002
                                 + 1.04113 w_k1_DIS_s002 + 1.031 w_k1_FDX_s002
                                 + 1.0985 w_k1_IBM_s002
                                 + 0.936073 w_k1_INTC_s002
                                 + 0.973509 w_k1_KO_s002
                                 + 1.05496 w_k1_MAT_s002
                                 + 0.986222 w_k1_MSFT_s002
                                 + 1.10932 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.18507 w_k1_ADM_s002
                                 + 1.65931 w_k1_AXP_s002
                                 + 1.64829 w_k1_DIS_s002
                                 + 1.35275 w_k1_FDX_s002 + 1.2662 w_k1_IBM_s002
                                 + 1.45401 w_k1_INTC_s002
                                 + 1.21999 w_k1_KO_s002 + 1.36374 w_k1_MAT_s002
                                 + 1.36055 w_k1_MSFT_s002
                                 + 1.12572 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.88442 w_k1_ADM_s003
                                 + 1.08028 w_k1_AXP_s003 + 1.136 w_k1_DIS_s003
                                 + 1.07886 w_k1_FDX_s003
                                 + 0.883089 w_k1_IBM_s003
                                 + 0.964865 w_k1_INTC_s003
                                 + 1.12254 w_k1_KO_s003
                                 + 0.734786 w_k1_MAT_s003
                                 + 1.05166 w_k1_MSFT_s003
                                 + 1.00144 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.995606 w_k1_ADM_s003
                                 + 0.908455 w_k1_AXP_s003
                                 + 0.848747 w_k1_DIS_s003
                                 + 0.776241 w_k1_FDX_s003
                                 + 1.08341 w_k1_IBM_s003
                                 + 0.852489 w_k1_INTC_s003
                                 + 1.38478 w_k1_KO_s003 + 1.05424 w_k1_MAT_s003
                                 + 0.945425 w_k1_MSFT_s003
                                 + 0.951086 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.988138 w_k1_ADM_s003
                                 + 0.626789 w_k1_AXP_s003
                                 + 0.659692 w_k1_DIS_s003
                                 + 0.719928 w_k1_FDX_s003
                                 + 0.617224 w_k1_IBM_s003
                                 + 0.76038 w_k1_INTC_s003
                                 + 0.822578 w_k1_KO_s003
                                 + 0.515899 w_k1_MAT_s003
                                 + 0.795526 w_k1_MSFT_s003
                                 + 0.854461 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.58098 w_k1_ADM_s003
                                 + 1.37086 w_k1_AXP_s003
                                 + 1.67485 w_k1_DIS_s003
                                 + 1.19964 w_k1_FDX_s003 + 1.6326 w_k1_IBM_s003
                                 + 1.33263 w_k1_INTC_s003
                                 + 1.85882 w_k1_KO_s003 + 1.40105 w_k1_MAT_s003
                                 + 1.22115 w_k1_MSFT_s003
                                 + 1.33723 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.947509 w_k1_ADM_s003
                                 + 0.810179 w_k1_AXP_s003
                                 + 0.901496 w_k1_DIS_s003
                                 + 0.862246 w_k1_FDX_s003
                                 + 0.826729 w_k1_IBM_s003
                                 + 0.873631 w_k1_INTC_s003
                                 + 0.944741 w_k1_KO_s003
                                 + 0.998186 w_k1_MAT_s003
                                 + 1.01636 w_k1_MSFT_s003
                                 + 0.987489 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.38743 w_k1_ADM_s003
                                 + 0.955436 w_k1_AXP_s003
                                 + 0.887004 w_k1_DIS_s003
                                 + 0.873833 w_k1_FDX_s003
                                 + 0.905194 w_k1_IBM_s003
                                 + 0.916721 w_k1_INTC_s003
                                 + 0.930817 w_k1_KO_s003
                                 + 0.757182 w_k1_MAT_s003
                                 + 0.844861 w_k1_MSFT_s003
                                 + 0.905838 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.12419 w_k1_ADM_s003
                                 + 1.27717 w_k1_AXP_s003
                                 + 1.08623 w_k1_DIS_s003
                                 + 1.02031 w_k1_FDX_s003
                                 + 1.30149 w_k1_IBM_s003
                                 + 1.11749 w_k1_INTC_s003
                                 + 1.17976 w_k1_KO_s003 + 1.33077 w_k1_MAT_s003
                                 + 1.07944 w_k1_MSFT_s003
                                 + 0.97947 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.34865 w_k1_ADM_s003
                                 + 1.12632 w_k1_AXP_s003
                                 + 1.65758 w_k1_DIS_s003
                                 + 1.22384 w_k1_FDX_s003
                                 + 1.01392 w_k1_IBM_s003
                                 + 1.03805 w_k1_INTC_s003
                                 + 1.45146 w_k1_KO_s003
                                 + 0.864887 w_k1_MAT_s003
                                 + 1.28216 w_k1_MSFT_s003
                                 + 1.22023 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.14962 w_k1_ADM_s004
                                 + 1.61907 w_k1_AXP_s004
                                 + 0.868191 w_k1_DIS_s004
                                 + 1.09593 w_k1_FDX_s004
                                 + 1.07607 w_k1_IBM_s004
                                 + 1.21441 w_k1_INTC_s004 + 1.1321 w_k1_KO_s004
                                 + 1.36751 w_k1_MAT_s004
                                 + 1.09607 w_k1_MSFT_s004
                                 + 1.22755 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.15444 w_k1_ADM_s004
                                 + 1.28018 w_k1_AXP_s004
                                 + 0.989686 w_k1_DIS_s004
                                 + 1.04509 w_k1_FDX_s004
                                 + 0.696909 w_k1_IBM_s004
                                 + 0.974572 w_k1_INTC_s004
                                 + 1.0291 w_k1_KO_s004 + 1.14599 w_k1_MAT_s004
                                 + 0.890159 w_k1_MSFT_s004
                                 + 0.873235 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.927122 w_k1_ADM_s004
                                 + 0.870118 w_k1_AXP_s004
                                 + 1.26802 w_k1_DIS_s004
                                 + 1.11522 w_k1_FDX_s004
                                 + 1.07494 w_k1_IBM_s004
                                 + 1.10079 w_k1_INTC_s004
                                 + 1.27906 w_k1_KO_s004
                                 + 0.955462 w_k1_MAT_s004
                                 + 1.15123 w_k1_MSFT_s004
                                 + 1.18144 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.804313 w_k1_ADM_s004
                                 + 0.869949 w_k1_AXP_s004
                                 + 0.721137 w_k1_DIS_s004
                                 + 0.809198 w_k1_FDX_s004
                                 + 1.19804 w_k1_IBM_s004
                                 + 0.915145 w_k1_INTC_s004
                                 + 1.01915 w_k1_KO_s004 + 1.38091 w_k1_MAT_s004
                                 + 0.819435 w_k1_MSFT_s004
                                 + 1.00971 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09915 w_k1_ADM_s004
                                 + 2.05743 w_k1_AXP_s004
                                 + 1.47985 w_k1_DIS_s004
                                 + 1.44037 w_k1_FDX_s004
                                 + 1.25726 w_k1_IBM_s004
                                 + 1.11464 w_k1_INTC_s004
                                 + 1.31696 w_k1_KO_s004 + 2.01498 w_k1_MAT_s004
                                 + 1.05195 w_k1_MSFT_s004
                                 + 1.19919 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.645809 w_k1_ADM_s004
                                 + 0.765103 w_k1_AXP_s004
                                 + 0.76773 w_k1_DIS_s004
                                 + 0.777321 w_k1_FDX_s004
                                 + 0.743834 w_k1_IBM_s004
                                 + 0.801411 w_k1_INTC_s004
                                 + 0.87586 w_k1_KO_s004
                                 + 0.780389 w_k1_MAT_s004
                                 + 0.77995 w_k1_MSFT_s004
                                 + 0.801558 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.736199 w_k1_ADM_s004
                                 + 1.29375 w_k1_AXP_s004
                                 + 0.914493 w_k1_DIS_s004
                                 + 0.933231 w_k1_FDX_s004
                                 + 1.27722 w_k1_IBM_s004
                                 + 1.20562 w_k1_INTC_s004
                                 + 1.13713 w_k1_KO_s004 + 1.59999 w_k1_MAT_s004
                                 + 0.878512 w_k1_MSFT_s004
                                 + 1.12825 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.17763 w_k1_ADM_s004
                                 + 1.22201 w_k1_AXP_s004
                                 + 1.53069 w_k1_DIS_s004
                                 + 1.22434 w_k1_FDX_s004
                                 + 1.03112 w_k1_IBM_s004
                                 + 1.35859 w_k1_INTC_s004
                                 + 1.35018 w_k1_KO_s004
                                 + 0.903749 w_k1_MAT_s004
                                 + 1.16338 w_k1_MSFT_s004
                                 + 1.20275 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.36979 w_k1_ADM_s005
                                 + 1.26647 w_k1_AXP_s005
                                 + 1.48863 w_k1_DIS_s005
                                 + 1.34358 w_k1_FDX_s005
                                 + 0.889852 w_k1_IBM_s005
                                 + 1.46289 w_k1_INTC_s005
                                 + 1.05646 w_k1_KO_s005
                                 + 0.719654 w_k1_MAT_s005
                                 + 1.36909 w_k1_MSFT_s005
                                 + 1.02079 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.17199 w_k1_ADM_s005
                                 + 0.874699 w_k1_AXP_s005
                                 + 1.31744 w_k1_DIS_s005 + 1.0826 w_k1_FDX_s005
                                 + 0.913619 w_k1_IBM_s005
                                 + 1.32023 w_k1_INTC_s005 + 1.1997 w_k1_KO_s005
                                 + 0.627987 w_k1_MAT_s005
                                 + 1.27364 w_k1_MSFT_s005
                                 + 0.99475 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.91013 w_k1_ADM_s005
                                 + 1.20102 w_k1_AXP_s005
                                 + 0.927464 w_k1_DIS_s005
                                 + 1.03578 w_k1_FDX_s005
                                 + 0.92927 w_k1_IBM_s005
                                 + 1.1599 w_k1_INTC_s005
                                 + 0.927895 w_k1_KO_s005
                                 + 1.01501 w_k1_MAT_s005
                                 + 0.987597 w_k1_MSFT_s005
                                 + 1.00805 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.25302 w_k1_ADM_s005
                                 + 1.28819 w_k1_AXP_s005
                                 + 1.45539 w_k1_DIS_s005
                                 + 1.25164 w_k1_FDX_s005 + 1.45 w_k1_IBM_s005
                                 + 1.35674 w_k1_INTC_s005
                                 + 1.24981 w_k1_KO_s005 + 1.34287 w_k1_MAT_s005
                                 + 1.17981 w_k1_MSFT_s005
                                 + 0.95474 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.806576 w_k1_ADM_s005
                                 + 0.810315 w_k1_AXP_s005
                                 + 0.844562 w_k1_DIS_s005
                                 + 1.01286 w_k1_FDX_s005
                                 + 0.673737 w_k1_IBM_s005
                                 + 1.17776 w_k1_INTC_s005
                                 + 0.764515 w_k1_KO_s005
                                 + 0.604616 w_k1_MAT_s005
                                 + 1.01258 w_k1_MSFT_s005
                                 + 0.791266 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.954191 w_k1_ADM_s005
                                 + 1.00683 w_k1_AXP_s005
                                 + 1.11819 w_k1_DIS_s005
                                 + 0.929668 w_k1_FDX_s005
                                 + 1.58773 w_k1_IBM_s005
                                 + 1.55269 w_k1_INTC_s005 + 1.4228 w_k1_KO_s005
                                 + 1.23414 w_k1_MAT_s005
                                 + 1.21623 w_k1_MSFT_s005
                                 + 1.01924 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.979638 w_k1_ADM_s005
                                 + 0.921671 w_k1_AXP_s005
                                 + 1.20411 w_k1_DIS_s005
                                 + 1.10081 w_k1_FDX_s005
                                 + 1.20083 w_k1_IBM_s005
                                 + 1.30267 w_k1_INTC_s005
                                 + 1.14162 w_k1_KO_s005
                                 + 0.946111 w_k1_MAT_s005
                                 + 1.04636 w_k1_MSFT_s005
                                 + 1.11034 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.777586 w_k1_ADM_s005
                                 + 0.77072 w_k1_AXP_s005
                                 + 0.701818 w_k1_DIS_s005
                                 + 0.744875 w_k1_FDX_s005
                                 + 0.912811 w_k1_IBM_s005
                                 + 0.944295 w_k1_INTC_s005
                                 + 0.847772 w_k1_KO_s005
                                 + 0.839149 w_k1_MAT_s005
                                 + 0.859996 w_k1_MSFT_s005
                                 + 0.788947 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.860433 w_k1_ADM_s006
                                 + 1.31269 w_k1_AXP_s006
                                 + 1.06847 w_k1_DIS_s006
                                 + 0.954968 w_k1_FDX_s006
                                 + 1.41013 w_k1_IBM_s006
                                 + 1.2049 w_k1_INTC_s006 + 1.1132 w_k1_KO_s006
                                 + 1.11661 w_k1_MAT_s006
                                 + 0.931408 w_k1_MSFT_s006
                                 + 1.22086 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.92897 w_k1_ADM_s006
                                 + 1.24147 w_k1_AXP_s006
                                 + 0.782654 w_k1_DIS_s006
                                 + 0.714915 w_k1_FDX_s006
                                 + 1.71223 w_k1_IBM_s006
                                 + 0.723571 w_k1_INTC_s006
                                 + 0.971716 w_k1_KO_s006
                                 + 1.40724 w_k1_MAT_s006
                                 + 0.627293 w_k1_MSFT_s006
                                 + 0.891749 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.916722 w_k1_ADM_s006
                                 + 0.760938 w_k1_AXP_s006
                                 + 0.702082 w_k1_DIS_s006
                                 + 0.827044 w_k1_FDX_s006
                                 + 0.892258 w_k1_IBM_s006
                                 + 0.744879 w_k1_INTC_s006
                                 + 0.796217 w_k1_KO_s006
                                 + 1.06608 w_k1_MAT_s006
                                 + 0.794382 w_k1_MSFT_s006
                                 + 0.921543 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.43031 w_k1_ADM_s006
                                 + 1.78549 w_k1_AXP_s006
                                 + 1.55405 w_k1_DIS_s006
                                 + 1.68208 w_k1_FDX_s006
                                 + 1.18383 w_k1_IBM_s006
                                 + 1.35975 w_k1_INTC_s006 + 1.2894 w_k1_KO_s006
                                 + 2.23456 w_k1_MAT_s006
                                 + 1.34278 w_k1_MSFT_s006 + 1.4576 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.913438 w_k1_ADM_s006
                                 + 1.10508 w_k1_AXP_s006
                                 + 1.03475 w_k1_DIS_s006
                                 + 1.11514 w_k1_FDX_s006
                                 + 0.872326 w_k1_IBM_s006
                                 + 1.24831 w_k1_INTC_s006
                                 + 0.955088 w_k1_KO_s006
                                 + 1.02283 w_k1_MAT_s006
                                 + 1.18207 w_k1_MSFT_s006
                                 + 0.978281 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.825671 w_k1_ADM_s006
                                 + 0.706905 w_k1_AXP_s006
                                 + 0.615418 w_k1_DIS_s006
                                 + 0.668416 w_k1_FDX_s006
                                 + 0.958982 w_k1_IBM_s006
                                 + 0.621081 w_k1_INTC_s006
                                 + 0.80482 w_k1_KO_s006
                                 + 0.695915 w_k1_MAT_s006
                                 + 0.614401 w_k1_MSFT_s006
                                 + 0.856829 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.35183 w_k1_ADM_s006
                                 + 1.14068 w_k1_AXP_s006
                                 + 1.52016 w_k1_DIS_s006
                                 + 1.19087 w_k1_FDX_s006
                                 + 1.32231 w_k1_IBM_s006
                                 + 1.34391 w_k1_INTC_s006
                                 + 1.46813 w_k1_KO_s006 + 1.29707 w_k1_MAT_s006
                                 + 1.29697 w_k1_MSFT_s006
                                 + 1.20451 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.739777 w_k1_ADM_s006
                                 + 0.666723 w_k1_AXP_s006
                                 + 0.662242 w_k1_DIS_s006
                                 + 0.770571 w_k1_FDX_s006
                                 + 0.527524 w_k1_IBM_s006
                                 + 0.803073 w_k1_INTC_s006
                                 + 0.584768 w_k1_KO_s006
                                 + 0.388547 w_k1_MAT_s006
                                 + 0.726578 w_k1_MSFT_s006
                                 + 0.74441 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.874293 w_k1_ADM_s007
                                 + 0.755874 w_k1_AXP_s007
                                 + 0.887939 w_k1_DIS_s007
                                 + 0.809907 w_k1_FDX_s007
                                 + 1.26565 w_k1_IBM_s007
                                 + 0.881309 w_k1_INTC_s007
                                 + 0.931565 w_k1_KO_s007
                                 + 0.616588 w_k1_MAT_s007
                                 + 1.0085 w_k1_MSFT_s007
                                 + 0.950575 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.38586 w_k1_ADM_s007
                                 + 1.16405 w_k1_AXP_s007
                                 + 1.40023 w_k1_DIS_s007
                                 + 1.38252 w_k1_FDX_s007
                                 + 0.788574 w_k1_IBM_s007
                                 + 1.04205 w_k1_INTC_s007 + 1.1339 w_k1_KO_s007
                                 + 0.787742 w_k1_MAT_s007
                                 + 1.29003 w_k1_MSFT_s007
                                 + 1.09907 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.63596 w_k1_ADM_s007
                                 + 1.17416 w_k1_AXP_s007
                                 + 1.65708 w_k1_DIS_s007
                                 + 1.33156 w_k1_FDX_s007
                                 + 1.27103 w_k1_IBM_s007
                                 + 1.39637 w_k1_INTC_s007 + 1.4474 w_k1_KO_s007
                                 + 0.972556 w_k1_MAT_s007
                                 + 1.55365 w_k1_MSFT_s007
                                 + 1.19952 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.865409 w_k1_ADM_s007
                                 + 0.976909 w_k1_AXP_s007
                                 + 0.871134 w_k1_DIS_s007
                                 + 0.959916 w_k1_FDX_s007
                                 + 0.663429 w_k1_IBM_s007
                                 + 0.757794 w_k1_INTC_s007
                                 + 0.744412 w_k1_KO_s007
                                 + 0.619651 w_k1_MAT_s007
                                 + 0.96446 w_k1_MSFT_s007
                                 + 0.988948 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.91615 w_k1_ADM_s007
                                 + 0.764736 w_k1_AXP_s007
                                 + 0.879222 w_k1_DIS_s007
                                 + 0.815484 w_k1_FDX_s007
                                 + 0.783432 w_k1_IBM_s007
                                 + 0.788621 w_k1_INTC_s007
                                 + 0.893056 w_k1_KO_s007
                                 + 0.625527 w_k1_MAT_s007
                                 + 0.966998 w_k1_MSFT_s007
                                 + 0.924935 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.862912 w_k1_ADM_s007
                                 + 1.19923 w_k1_AXP_s007
                                 + 1.12192 w_k1_DIS_s007
                                 + 1.04946 w_k1_FDX_s007
                                 + 1.32057 w_k1_IBM_s007
                                 + 1.02017 w_k1_INTC_s007
                                 + 1.21668 w_k1_KO_s007 + 1.6139 w_k1_MAT_s007
                                 + 1.19488 w_k1_MSFT_s007 + 1.0927 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.23474 w_k1_ADM_s007
                                 + 0.939484 w_k1_AXP_s007
                                 + 1.24665 w_k1_DIS_s007
                                 + 1.08235 w_k1_FDX_s007
                                 + 0.900801 w_k1_IBM_s007
                                 + 1.10127 w_k1_INTC_s007
                                 + 1.16119 w_k1_KO_s007
                                 + 0.900903 w_k1_MAT_s007
                                 + 1.23825 w_k1_MSFT_s007
                                 + 1.04084 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.64789 w_k1_ADM_s007
                                 + 0.752004 w_k1_AXP_s007
                                 + 0.599528 w_k1_DIS_s007
                                 + 0.696101 w_k1_FDX_s007
                                 + 0.855586 w_k1_IBM_s007
                                 + 0.827443 w_k1_INTC_s007
                                 + 0.708436 w_k1_KO_s007
                                 + 0.617345 w_k1_MAT_s007
                                 + 0.805258 w_k1_MSFT_s007
                                 + 0.794114 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.687808 w_k1_ADM_s010
                                 + 0.688801 w_k1_AXP_s010
                                 + 0.66381 w_k1_DIS_s010
                                 + 0.700945 w_k1_FDX_s010
                                 + 0.859294 w_k1_IBM_s010
                                 + 0.767461 w_k1_INTC_s010
                                 + 0.719744 w_k1_KO_s010
                                 + 0.782177 w_k1_MAT_s010
                                 + 0.700405 w_k1_MSFT_s010
                                 + 0.772696 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.20921 w_k1_ADM_s010
                                 + 1.35149 w_k1_AXP_s010
                                 + 1.50298 w_k1_DIS_s010
                                 + 1.17938 w_k1_FDX_s010
                                 + 1.44768 w_k1_IBM_s010
                                 + 0.972928 w_k1_INTC_s010
                                 + 1.46789 w_k1_KO_s010 + 2.04529 w_k1_MAT_s010
                                 + 1.07663 w_k1_MSFT_s010 + 1.1725 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.953398 w_k1_ADM_s010
                                 + 0.866372 w_k1_AXP_s010
                                 + 1.00793 w_k1_DIS_s010
                                 + 0.913747 w_k1_FDX_s010
                                 + 0.993961 w_k1_IBM_s010
                                 + 0.977586 w_k1_INTC_s010
                                 + 0.88802 w_k1_KO_s010
                                 + 0.705563 w_k1_MAT_s010
                                 + 0.921644 w_k1_MSFT_s010
                                 + 0.916361 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.12682 w_k1_ADM_s010
                                 + 1.22795 w_k1_AXP_s010
                                 + 1.33275 w_k1_DIS_s010
                                 + 1.35011 w_k1_FDX_s010
                                 + 0.943293 w_k1_IBM_s010
                                 + 1.44285 w_k1_INTC_s010
                                 + 1.07649 w_k1_KO_s010 + 1.00823 w_k1_MAT_s010
                                 + 1.26846 w_k1_MSFT_s010
                                 + 1.11935 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.03427 w_k1_ADM_s010
                                 + 1.21003 w_k1_AXP_s010
                                 + 0.82876 w_k1_DIS_s010
                                 + 1.00548 w_k1_FDX_s010
                                 + 0.991813 w_k1_IBM_s010
                                 + 0.992882 w_k1_INTC_s010
                                 + 0.94323 w_k1_KO_s010 + 1.17633 w_k1_MAT_s010
                                 + 0.948637 w_k1_MSFT_s010
                                 + 0.938193 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.864974 w_k1_ADM_s010
                                 + 0.840527 w_k1_AXP_s010
                                 + 0.744386 w_k1_DIS_s010
                                 + 0.923296 w_k1_FDX_s010
                                 + 0.673846 w_k1_IBM_s010
                                 + 0.791452 w_k1_INTC_s010
                                 + 0.676947 w_k1_KO_s010
                                 + 0.661491 w_k1_MAT_s010
                                 + 0.895657 w_k1_MSFT_s010
                                 + 0.755914 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.981105 w_k1_ADM_s010
                                 + 0.891162 w_k1_AXP_s010
                                 + 0.884436 w_k1_DIS_s010
                                 + 1.14145 w_k1_FDX_s010
                                 + 0.512164 w_k1_IBM_s010
                                 + 0.985007 w_k1_INTC_s010
                                 + 0.647075 w_k1_KO_s010
                                 + 0.490286 w_k1_MAT_s010
                                 + 0.991861 w_k1_MSFT_s010
                                 + 0.777909 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.869785 w_k1_ADM_s010
                                 + 1.02978 w_k1_AXP_s010
                                 + 0.84597 w_k1_DIS_s010
                                 + 0.821707 w_k1_FDX_s010
                                 + 1.52996 w_k1_IBM_s010
                                 + 0.771044 w_k1_INTC_s010
                                 + 0.897497 w_k1_KO_s010
                                 + 1.39231 w_k1_MAT_s010
                                 + 0.775027 w_k1_MSFT_s010
                                 + 0.901615 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.829869 w_k1_ADM_s011
                                 + 0.845415 w_k1_AXP_s011
                                 + 0.831055 w_k1_DIS_s011
                                 + 0.822444 w_k1_FDX_s011
                                 + 0.874267 w_k1_IBM_s011
                                 + 0.859538 w_k1_INTC_s011
                                 + 0.869902 w_k1_KO_s011
                                 + 1.06404 w_k1_MAT_s011
                                 + 0.893466 w_k1_MSFT_s011
                                 + 0.836287 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.858422 w_k1_ADM_s011
                                 + 1.07344 w_k1_AXP_s011
                                 + 0.885976 w_k1_DIS_s011
                                 + 1.04996 w_k1_FDX_s011
                                 + 0.821816 w_k1_IBM_s011
                                 + 0.92802 w_k1_INTC_s011
                                 + 0.876762 w_k1_KO_s011 + 1.0174 w_k1_MAT_s011
                                 + 1.025 w_k1_MSFT_s011 + 0.979566 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.33396 w_k1_ADM_s011
                                 + 1.39305 w_k1_AXP_s011 + 1.2282 w_k1_DIS_s011
                                 + 1.16852 w_k1_FDX_s011
                                 + 1.16449 w_k1_IBM_s011
                                 + 1.04556 w_k1_INTC_s011
                                 + 1.15356 w_k1_KO_s011 + 1.269 w_k1_MAT_s011
                                 + 1.09985 w_k1_MSFT_s011
                                 + 1.08852 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.02189 w_k1_ADM_s011
                                 + 1.03864 w_k1_AXP_s011
                                 + 1.03856 w_k1_DIS_s011
                                 + 0.920806 w_k1_FDX_s011
                                 + 1.36557 w_k1_IBM_s011
                                 + 0.717802 w_k1_INTC_s011
                                 + 1.23587 w_k1_KO_s011 + 1.71493 w_k1_MAT_s011
                                 + 0.918136 w_k1_MSFT_s011
                                 + 0.968573 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.736726 w_k1_ADM_s011
                                 + 0.846453 w_k1_AXP_s011
                                 + 0.703707 w_k1_DIS_s011
                                 + 0.811488 w_k1_FDX_s011
                                 + 0.616142 w_k1_IBM_s011
                                 + 0.779671 w_k1_INTC_s011
                                 + 0.808232 w_k1_KO_s011
                                 + 0.703781 w_k1_MAT_s011
                                 + 0.782675 w_k1_MSFT_s011 + 0.805 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.930149 w_k1_ADM_s011
                                 + 1.31804 w_k1_AXP_s011
                                 + 1.68772 w_k1_DIS_s011
                                 + 1.36922 w_k1_FDX_s011
                                 + 1.14696 w_k1_IBM_s011
                                 + 1.25598 w_k1_INTC_s011
                                 + 1.28105 w_k1_KO_s011 + 1.49621 w_k1_MAT_s011
                                 + 1.3235 w_k1_MSFT_s011 + 1.13715 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.919796 w_k1_ADM_s011
                                 + 0.784521 w_k1_AXP_s011
                                 + 1.02513 w_k1_DIS_s011
                                 + 0.993981 w_k1_FDX_s011
                                 + 0.696739 w_k1_IBM_s011
                                 + 0.961226 w_k1_INTC_s011
                                 + 0.998263 w_k1_KO_s011
                                 + 0.937529 w_k1_MAT_s011
                                 + 1.2314 w_k1_MSFT_s011
                                 + 0.995225 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.89483 w_k1_ADM_s011
                                 + 1.53401 w_k1_AXP_s011
                                 + 0.910889 w_k1_DIS_s011
                                 + 0.968684 w_k1_FDX_s011
                                 + 1.2062 w_k1_IBM_s011
                                 + 0.664577 w_k1_INTC_s011
                                 + 0.856243 w_k1_KO_s011
                                 + 1.95959 w_k1_MAT_s011
                                 + 0.84092 w_k1_MSFT_s011
                                 + 0.855219 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.832222 w_k1_ADM_s012
                                 + 0.863735 w_k1_AXP_s012
                                 + 0.87375 w_k1_DIS_s012
                                 + 0.955925 w_k1_FDX_s012
                                 + 0.641451 w_k1_IBM_s012
                                 + 0.891804 w_k1_INTC_s012
                                 + 0.744971 w_k1_KO_s012
                                 + 0.520683 w_k1_MAT_s012
                                 + 0.866143 w_k1_MSFT_s012
                                 + 1.05952 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.22701 w_k1_ADM_s012
                                 + 1.12905 w_k1_AXP_s012
                                 + 0.839427 w_k1_DIS_s012
                                 + 0.987166 w_k1_FDX_s012
                                 + 1.02831 w_k1_IBM_s012
                                 + 0.853964 w_k1_INTC_s012
                                 + 0.878456 w_k1_KO_s012
                                 + 1.16297 w_k1_MAT_s012
                                 + 0.902633 w_k1_MSFT_s012
                                 + 1.08414 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.25671 w_k1_ADM_s012
                                 + 1.39672 w_k1_AXP_s012
                                 + 1.36064 w_k1_DIS_s012
                                 + 1.29372 w_k1_FDX_s012
                                 + 1.45897 w_k1_IBM_s012
                                 + 0.932489 w_k1_INTC_s012
                                 + 1.23968 w_k1_KO_s012 + 1.52071 w_k1_MAT_s012
                                 + 1.08686 w_k1_MSFT_s012
                                 + 1.47794 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.56943 w_k1_ADM_s012
                                 + 1.34416 w_k1_AXP_s012
                                 + 1.64497 w_k1_DIS_s012
                                 + 1.45351 w_k1_FDX_s012
                                 + 1.03014 w_k1_IBM_s012
                                 + 1.08856 w_k1_INTC_s012 + 1.1003 w_k1_KO_s012
                                 + 1.11281 w_k1_MAT_s012
                                 + 1.33094 w_k1_MSFT_s012 + 1.3455 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.966496 w_k1_ADM_s012
                                 + 0.901702 w_k1_AXP_s012
                                 + 0.965032 w_k1_DIS_s012
                                 + 0.926681 w_k1_FDX_s012 + 1.42 w_k1_IBM_s012
                                 + 0.906888 w_k1_INTC_s012
                                 + 1.00451 w_k1_KO_s012 + 1.01082 w_k1_MAT_s012
                                 + 0.933783 w_k1_MSFT_s012
                                 + 1.05985 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.28665 w_k1_ADM_s012
                                 + 0.929317 w_k1_AXP_s012
                                 + 1.18125 w_k1_DIS_s012
                                 + 0.870669 w_k1_FDX_s012
                                 + 1.96365 w_k1_IBM_s012
                                 + 0.876565 w_k1_INTC_s012
                                 + 1.44191 w_k1_KO_s012 + 1.566 w_k1_MAT_s012
                                 + 0.871039 w_k1_MSFT_s012
                                 + 1.16649 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.86479 w_k1_ADM_s012
                                 + 0.676066 w_k1_AXP_s012
                                 + 0.625319 w_k1_DIS_s012
                                 + 0.851484 w_k1_FDX_s012
                                 + 0.82027 w_k1_IBM_s012
                                 + 0.804889 w_k1_INTC_s012
                                 + 0.692345 w_k1_KO_s012
                                 + 0.807519 w_k1_MAT_s012
                                 + 0.773268 w_k1_MSFT_s012
                                 + 0.89026 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.03148 w_k1_ADM_s012
                                 + 0.825731 w_k1_AXP_s012
                                 + 1.21405 w_k1_DIS_s012
                                 + 1.08448 w_k1_FDX_s012
                                 + 0.854578 w_k1_IBM_s012
                                 + 1.03921 w_k1_INTC_s012
                                 + 1.02002 w_k1_KO_s012
                                 + 0.799872 w_k1_MAT_s012
                                 + 1.14851 w_k1_MSFT_s012
                                 + 1.15177 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.997465 w_k1_ADM_s013
                                 + 1.01708 w_k1_AXP_s013
                                 + 0.929456 w_k1_DIS_s013
                                 + 0.895277 w_k1_FDX_s013
                                 + 0.941351 w_k1_IBM_s013
                                 + 0.690189 w_k1_INTC_s013
                                 + 0.903527 w_k1_KO_s013
                                 + 1.35034 w_k1_MAT_s013
                                 + 0.86864 w_k1_MSFT_s013
                                 + 0.861769 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.612523 w_k1_ADM_s013
                                 + 0.70603 w_k1_AXP_s013
                                 + 0.955073 w_k1_DIS_s013
                                 + 0.822045 w_k1_FDX_s013
                                 + 0.958153 w_k1_IBM_s013
                                 + 0.796385 w_k1_INTC_s013
                                 + 1.0183 w_k1_KO_s013 + 0.749932 w_k1_MAT_s013
                                 + 0.814847 w_k1_MSFT_s013
                                 + 0.903275 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.962938 w_k1_ADM_s013
                                 + 1.36285 w_k1_AXP_s013
                                 + 1.09999 w_k1_DIS_s013
                                 + 1.12184 w_k1_FDX_s013
                                 + 1.37533 w_k1_IBM_s013
                                 + 1.01054 w_k1_INTC_s013
                                 + 1.34821 w_k1_KO_s013 + 1.94014 w_k1_MAT_s013
                                 + 1.07484 w_k1_MSFT_s013
                                 + 1.27364 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.976348 w_k1_ADM_s013
                                 + 0.800649 w_k1_AXP_s013
                                 + 0.732394 w_k1_DIS_s013
                                 + 0.886721 w_k1_FDX_s013
                                 + 0.647586 w_k1_IBM_s013
                                 + 0.661458 w_k1_INTC_s013
                                 + 0.701147 w_k1_KO_s013
                                 + 0.691211 w_k1_MAT_s013
                                 + 0.824842 w_k1_MSFT_s013
                                 + 0.848342 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.921276 w_k1_ADM_s013
                                 + 1.21824 w_k1_AXP_s013
                                 + 1.28402 w_k1_DIS_s013
                                 + 1.17889 w_k1_FDX_s013
                                 + 0.893785 w_k1_IBM_s013
                                 + 0.943863 w_k1_INTC_s013
                                 + 0.86723 w_k1_KO_s013 + 1.03627 w_k1_MAT_s013
                                 + 1.02946 w_k1_MSFT_s013
                                 + 0.98133 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.00952 w_k1_ADM_s013
                                 + 1.02448 w_k1_AXP_s013
                                 + 1.03825 w_k1_DIS_s013
                                 + 1.11193 w_k1_FDX_s013
                                 + 1.13083 w_k1_IBM_s013
                                 + 0.870923 w_k1_INTC_s013
                                 + 1.20568 w_k1_KO_s013 + 1.15292 w_k1_MAT_s013
                                 + 1.00891 w_k1_MSFT_s013
                                 + 1.08012 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.87482 w_k1_ADM_s013
                                 + 0.720291 w_k1_AXP_s013
                                 + 0.687442 w_k1_DIS_s013
                                 + 0.705534 w_k1_FDX_s013
                                 + 0.892948 w_k1_IBM_s013
                                 + 0.804834 w_k1_INTC_s013
                                 + 0.842229 w_k1_KO_s013
                                 + 0.75515 w_k1_MAT_s013
                                 + 0.798915 w_k1_MSFT_s013
                                 + 0.850417 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.03384 w_k1_ADM_s013
                                 + 1.60779 w_k1_AXP_s013
                                 + 1.65629 w_k1_DIS_s013
                                 + 1.44754 w_k1_FDX_s013
                                 + 1.56004 w_k1_IBM_s013
                                 + 1.2041 w_k1_INTC_s013 + 1.22968 w_k1_KO_s013
                                 + 1.92206 w_k1_MAT_s013
                                 + 1.1582 w_k1_MSFT_s013 + 1.16507 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.921541 w_k1_ADM_s014
                                 + 0.756213 w_k1_AXP_s014
                                 + 0.653028 w_k1_DIS_s014
                                 + 0.739912 w_k1_FDX_s014
                                 + 0.938444 w_k1_IBM_s014
                                 + 0.867751 w_k1_INTC_s014
                                 + 0.948041 w_k1_KO_s014
                                 + 1.19869 w_k1_MAT_s014
                                 + 0.728482 w_k1_MSFT_s014
                                 + 0.831488 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.07306 w_k1_ADM_s014
                                 + 0.995322 w_k1_AXP_s014
                                 + 0.80097 w_k1_DIS_s014
                                 + 0.887682 w_k1_FDX_s014
                                 + 0.984686 w_k1_IBM_s014
                                 + 1.07333 w_k1_INTC_s014
                                 + 1.01953 w_k1_KO_s014 + 1.49447 w_k1_MAT_s014
                                 + 0.967823 w_k1_MSFT_s014
                                 + 0.9812 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.3901 w_k1_ADM_s014
                                 + 1.26533 w_k1_AXP_s014
                                 + 1.70069 w_k1_DIS_s014
                                 + 1.29135 w_k1_FDX_s014
                                 + 1.24247 w_k1_IBM_s014
                                 + 1.71885 w_k1_INTC_s014
                                 + 1.71995 w_k1_KO_s014 + 1.44533 w_k1_MAT_s014
                                 + 1.43747 w_k1_MSFT_s014
                                 + 1.22251 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.17387 w_k1_ADM_s014
                                 + 0.957165 w_k1_AXP_s014
                                 + 1.10762 w_k1_DIS_s014
                                 + 1.03987 w_k1_FDX_s014
                                 + 0.863061 w_k1_IBM_s014
                                 + 1.02766 w_k1_INTC_s014
                                 + 1.15771 w_k1_KO_s014
                                 + 0.963631 w_k1_MAT_s014
                                 + 1.07884 w_k1_MSFT_s014
                                 + 1.13073 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.20604 w_k1_ADM_s014
                                 + 1.29071 w_k1_AXP_s014
                                 + 1.28697 w_k1_DIS_s014 + 1.1474 w_k1_FDX_s014
                                 + 1.10029 w_k1_IBM_s014
                                 + 1.21132 w_k1_INTC_s014
                                 + 1.40017 w_k1_KO_s014 + 1.75451 w_k1_MAT_s014
                                 + 1.11855 w_k1_MSFT_s014
                                 + 1.12788 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.803648 w_k1_ADM_s014
                                 + 0.605987 w_k1_AXP_s014
                                 + 0.666715 w_k1_DIS_s014
                                 + 0.737994 w_k1_FDX_s014
                                 + 0.714693 w_k1_IBM_s014
                                 + 0.846277 w_k1_INTC_s014
                                 + 0.762518 w_k1_KO_s014
                                 + 0.932644 w_k1_MAT_s014
                                 + 0.745588 w_k1_MSFT_s014
                                 + 0.793206 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.925512 w_k1_ADM_s014
                                 + 0.968846 w_k1_AXP_s014
                                 + 0.798257 w_k1_DIS_s014
                                 + 0.919326 w_k1_FDX_s014
                                 + 0.696833 w_k1_IBM_s014
                                 + 0.967386 w_k1_INTC_s014
                                 + 0.784033 w_k1_KO_s014
                                 + 0.984207 w_k1_MAT_s014
                                 + 0.765129 w_k1_MSFT_s014
                                 + 0.878666 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.828868 w_k1_ADM_s014
                                 + 0.731358 w_k1_AXP_s014
                                 + 1.02462 w_k1_DIS_s014
                                 + 0.804789 w_k1_FDX_s014
                                 + 1.29353 w_k1_IBM_s014
                                 + 1.15048 w_k1_INTC_s014
                                 + 1.31303 w_k1_KO_s014 + 1.33651 w_k1_MAT_s014
                                 + 0.943091 w_k1_MSFT_s014
                                 + 0.99545 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.25857 w_k1_ADM_s015
                                 + 1.15231 w_k1_AXP_s015 + 1.6566 w_k1_DIS_s015
                                 + 1.28811 w_k1_FDX_s015
                                 + 1.06886 w_k1_IBM_s015
                                 + 1.08206 w_k1_INTC_s015
                                 + 1.14131 w_k1_KO_s015
                                 + 0.946691 w_k1_MAT_s015
                                 + 1.23696 w_k1_MSFT_s015
                                 + 1.21812 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.782135 w_k1_ADM_s015
                                 + 0.688458 w_k1_AXP_s015
                                 + 0.868702 w_k1_DIS_s015
                                 + 0.854549 w_k1_FDX_s015
                                 + 0.71621 w_k1_IBM_s015
                                 + 0.962892 w_k1_INTC_s015
                                 + 0.713881 w_k1_KO_s015
                                 + 0.604069 w_k1_MAT_s015
                                 + 0.959058 w_k1_MSFT_s015
                                 + 0.872256 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.896783 w_k1_ADM_s015
                                 + 0.948145 w_k1_AXP_s015
                                 + 1.04202 w_k1_DIS_s015
                                 + 0.882307 w_k1_FDX_s015
                                 + 1.15893 w_k1_IBM_s015
                                 + 1.03302 w_k1_INTC_s015
                                 + 1.01634 w_k1_KO_s015
                                 + 0.924841 w_k1_MAT_s015
                                 + 0.822184 w_k1_MSFT_s015
                                 + 0.927627 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.933203 w_k1_ADM_s015
                                 + 1.19918 w_k1_AXP_s015
                                 + 1.02516 w_k1_DIS_s015
                                 + 1.10961 w_k1_FDX_s015
                                 + 0.846542 w_k1_IBM_s015
                                 + 1.06909 w_k1_INTC_s015
                                 + 0.890457 w_k1_KO_s015
                                 + 0.991129 w_k1_MAT_s015
                                 + 0.903286 w_k1_MSFT_s015
                                 + 0.945835 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.821882 w_k1_ADM_s015
                                 + 1.07269 w_k1_AXP_s015
                                 + 1.25709 w_k1_DIS_s015
                                 + 1.01096 w_k1_FDX_s015
                                 + 1.30952 w_k1_IBM_s015
                                 + 1.25601 w_k1_INTC_s015
                                 + 1.23008 w_k1_KO_s015 + 1.31189 w_k1_MAT_s015
                                 + 1.15672 w_k1_MSFT_s015
                                 + 0.908557 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.626143 w_k1_ADM_s015
                                 + 0.79246 w_k1_AXP_s015
                                 + 0.70866 w_k1_DIS_s015
                                 + 0.803028 w_k1_FDX_s015
                                 + 0.774518 w_k1_IBM_s015
                                 + 0.746189 w_k1_INTC_s015
                                 + 0.714157 w_k1_KO_s015
                                 + 0.707476 w_k1_MAT_s015
                                 + 0.801446 w_k1_MSFT_s015
                                 + 0.739859 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.924116 w_k1_ADM_s015
                                 + 1.57248 w_k1_AXP_s015
                                 + 1.57076 w_k1_DIS_s015
                                 + 1.38622 w_k1_FDX_s015
                                 + 1.56278 w_k1_IBM_s015
                                 + 1.45602 w_k1_INTC_s015
                                 + 1.13237 w_k1_KO_s015 + 1.80302 w_k1_MAT_s015
                                 + 1.23698 w_k1_MSFT_s015
                                 + 1.07554 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.88153 w_k1_ADM_s015
                                 + 0.808991 w_k1_AXP_s015
                                 + 0.844632 w_k1_DIS_s015
                                 + 0.842424 w_k1_FDX_s015
                                 + 1.0064 w_k1_IBM_s015
                                 + 0.775137 w_k1_INTC_s015
                                 + 0.799064 w_k1_KO_s015
                                 + 0.975294 w_k1_MAT_s015
                                 + 0.825172 w_k1_MSFT_s015
                                 + 0.912218 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.00374 w_k1_ADM_s016
                                 + 0.759716 w_k1_AXP_s016
                                 + 1.09572 w_k1_DIS_s016
                                 + 0.909745 w_k1_FDX_s016
                                 + 1.12806 w_k1_IBM_s016
                                 + 0.97494 w_k1_INTC_s016
                                 + 1.20281 w_k1_KO_s016
                                 + 0.892025 w_k1_MAT_s016
                                 + 1.05987 w_k1_MSFT_s016 + 0.9455 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.97045 w_k1_ADM_s016
                                 + 0.587293 w_k1_AXP_s016
                                 + 0.615292 w_k1_DIS_s016
                                 + 0.76919 w_k1_FDX_s016
                                 + 0.599925 w_k1_IBM_s016
                                 + 0.723503 w_k1_INTC_s016
                                 + 0.673464 w_k1_KO_s016
                                 + 0.520804 w_k1_MAT_s016
                                 + 0.746968 w_k1_MSFT_s016
                                 + 0.849144 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.25776 w_k1_ADM_s016
                                 + 0.975719 w_k1_AXP_s016
                                 + 1.10267 w_k1_DIS_s016
                                 + 1.11963 w_k1_FDX_s016
                                 + 0.91147 w_k1_IBM_s016
                                 + 1.05312 w_k1_INTC_s016
                                 + 0.842022 w_k1_KO_s016
                                 + 0.69079 w_k1_MAT_s016
                                 + 1.1841 w_k1_MSFT_s016
                                 + 0.840317 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.13845 w_k1_ADM_s016
                                 + 1.10638 w_k1_AXP_s016 + 1.1966 w_k1_DIS_s016
                                 + 1.14547 w_k1_FDX_s016
                                 + 0.739214 w_k1_IBM_s016
                                 + 1.07501 w_k1_INTC_s016
                                 + 1.03442 w_k1_KO_s016
                                 + 0.628661 w_k1_MAT_s016
                                 + 1.08673 w_k1_MSFT_s016
                                 + 0.981587 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.18173 w_k1_ADM_s016
                                 + 1.50902 w_k1_AXP_s016
                                 + 1.12695 w_k1_DIS_s016
                                 + 1.05944 w_k1_FDX_s016
                                 + 1.39224 w_k1_IBM_s016
                                 + 1.23196 w_k1_INTC_s016
                                 + 1.28255 w_k1_KO_s016 + 1.32194 w_k1_MAT_s016
                                 + 1.22628 w_k1_MSFT_s016
                                 + 1.05527 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.99503 w_k1_ADM_s016
                                 + 1.39756 w_k1_AXP_s016
                                 + 1.77895 w_k1_DIS_s016
                                 + 1.32747 w_k1_FDX_s016
                                 + 1.50899 w_k1_IBM_s016
                                 + 1.11777 w_k1_INTC_s016
                                 + 1.58836 w_k1_KO_s016 + 1.15679 w_k1_MAT_s016
                                 + 1.09175 w_k1_MSFT_s016
                                 + 1.30817 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.20414 w_k1_ADM_s016
                                 + 0.832216 w_k1_AXP_s016
                                 + 0.859463 w_k1_DIS_s016
                                 + 1.09818 w_k1_FDX_s016
                                 + 0.547995 w_k1_IBM_s016
                                 + 0.925918 w_k1_INTC_s016
                                 + 0.705166 w_k1_KO_s016
                                 + 0.351873 w_k1_MAT_s016
                                 + 1.05005 w_k1_MSFT_s016
                                 + 0.868201 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.08611 w_k1_ADM_s016
                                 + 0.760381 w_k1_AXP_s016
                                 + 0.813804 w_k1_DIS_s016
                                 + 0.839352 w_k1_FDX_s016
                                 + 0.710309 w_k1_IBM_s016
                                 + 0.908369 w_k1_INTC_s016
                                 + 0.869325 w_k1_KO_s016
                                 + 0.631474 w_k1_MAT_s016
                                 + 0.876533 w_k1_MSFT_s016
                                 + 0.929623 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.31765 w_k1_ADM_s017
                                 + 0.998072 w_k1_AXP_s017
                                 + 1.60336 w_k1_DIS_s017
                                 + 1.25687 w_k1_FDX_s017
                                 + 0.900947 w_k1_IBM_s017
                                 + 0.87196 w_k1_INTC_s017
                                 + 1.05932 w_k1_KO_s017
                                 + 0.838264 w_k1_MAT_s017
                                 + 1.17999 w_k1_MSFT_s017
                                 + 1.23384 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.942277 w_k1_ADM_s017
                                 + 0.936992 w_k1_AXP_s017
                                 + 0.99379 w_k1_DIS_s017
                                 + 0.805002 w_k1_FDX_s017
                                 + 1.35847 w_k1_IBM_s017
                                 + 0.712549 w_k1_INTC_s017
                                 + 1.10901 w_k1_KO_s017 + 1.32083 w_k1_MAT_s017
                                 + 0.851104 w_k1_MSFT_s017
                                 + 1.0704 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.603085 w_k1_ADM_s017
                                 + 0.653786 w_k1_AXP_s017
                                 + 0.55023 w_k1_DIS_s017
                                 + 0.663055 w_k1_FDX_s017
                                 + 1.07023 w_k1_IBM_s017
                                 + 0.69477 w_k1_INTC_s017
                                 + 0.78591 w_k1_KO_s017
                                 + 0.936365 w_k1_MAT_s017
                                 + 0.677362 w_k1_MSFT_s017
                                 + 0.739705 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.19665 w_k1_ADM_s017
                                 + 1.78733 w_k1_AXP_s017
                                 + 1.80301 w_k1_DIS_s017 + 1.3589 w_k1_FDX_s017
                                 + 1.62511 w_k1_IBM_s017
                                 + 1.09392 w_k1_INTC_s017
                                 + 1.47892 w_k1_KO_s017 + 2.45264 w_k1_MAT_s017
                                 + 1.26835 w_k1_MSFT_s017
                                 + 1.18126 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.81935 w_k1_ADM_s017
                                 + 0.795676 w_k1_AXP_s017
                                 + 0.873762 w_k1_DIS_s017
                                 + 0.814572 w_k1_FDX_s017
                                 + 1.04769 w_k1_IBM_s017
                                 + 0.729875 w_k1_INTC_s017
                                 + 0.85483 w_k1_KO_s017 + 1.14461 w_k1_MAT_s017
                                 + 0.859431 w_k1_MSFT_s017
                                 + 0.905266 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.708145 w_k1_ADM_s017
                                 + 0.8353 w_k1_AXP_s017
                                 + 0.683642 w_k1_DIS_s017
                                 + 0.772212 w_k1_FDX_s017
                                 + 0.734918 w_k1_IBM_s017
                                 + 0.791657 w_k1_INTC_s017
                                 + 0.67924 w_k1_KO_s017
                                 + 0.712906 w_k1_MAT_s017
                                 + 0.640928 w_k1_MSFT_s017
                                 + 0.856462 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.959484 w_k1_ADM_s017
                                 + 1.33802 w_k1_AXP_s017
                                 + 1.07668 w_k1_DIS_s017
                                 + 1.03295 w_k1_FDX_s017
                                 + 0.976812 w_k1_IBM_s017
                                 + 1.09653 w_k1_INTC_s017
                                 + 0.933175 w_k1_KO_s017
                                 + 1.20536 w_k1_MAT_s017
                                 + 1.01735 w_k1_MSFT_s017
                                 + 1.01047 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.777146 w_k1_ADM_s017
                                 + 0.811169 w_k1_AXP_s017
                                 + 0.899095 w_k1_DIS_s017
                                 + 0.983669 w_k1_FDX_s017
                                 + 0.625587 w_k1_IBM_s017
                                 + 0.731959 w_k1_INTC_s017
                                 + 0.632704 w_k1_KO_s017
                                 + 0.627832 w_k1_MAT_s017
                                 + 0.877605 w_k1_MSFT_s017
                                 + 0.750108 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.16647 w_k1_ADM_s100
                                 + 1.11045 w_k1_AXP_s100
                                 + 1.13341 w_k1_DIS_s100
                                 + 1.03694 w_k1_FDX_s100
                                 + 1.43413 w_k1_IBM_s100
                                 + 1.04552 w_k1_INTC_s100
                                 + 1.32448 w_k1_KO_s100 + 1.29816 w_k1_MAT_s100
                                 + 1.02908 w_k1_MSFT_s100
                                 + 1.13193 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.917545 w_k1_ADM_s100
                                 + 0.956938 w_k1_AXP_s100
                                 + 0.68036 w_k1_DIS_s100
                                 + 0.808771 w_k1_FDX_s100
                                 + 0.855816 w_k1_IBM_s100
                                 + 0.968056 w_k1_INTC_s100
                                 + 0.790574 w_k1_KO_s100
                                 + 0.804449 w_k1_MAT_s100
                                 + 0.761443 w_k1_MSFT_s100
                                 + 0.842632 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.24841 w_k1_ADM_s100
                                 + 0.879657 w_k1_AXP_s100
                                 + 0.991424 w_k1_DIS_s100
                                 + 0.885076 w_k1_FDX_s100
                                 + 1.01998 w_k1_IBM_s100
                                 + 1.03999 w_k1_INTC_s100
                                 + 1.15092 w_k1_KO_s100 + 1.01191 w_k1_MAT_s100
                                 + 0.937286 w_k1_MSFT_s100
                                 + 0.946044 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.29518 w_k1_ADM_s100
                                 + 1.38398 w_k1_AXP_s100
                                 + 1.45633 w_k1_DIS_s100
                                 + 1.30296 w_k1_FDX_s100
                                 + 1.05567 w_k1_IBM_s100
                                 + 1.23145 w_k1_INTC_s100
                                 + 1.25204 w_k1_KO_s100 + 1.0281 w_k1_MAT_s100
                                 + 1.17657 w_k1_MSFT_s100 + 1.2024 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09541 w_k1_ADM_s100
                                 + 1.17919 w_k1_AXP_s100
                                 + 1.07679 w_k1_DIS_s100
                                 + 1.14825 w_k1_FDX_s100
                                 + 0.706422 w_k1_IBM_s100
                                 + 1.20734 w_k1_INTC_s100
                                 + 0.823044 w_k1_KO_s100
                                 + 0.887083 w_k1_MAT_s100
                                 + 0.9972 w_k1_MSFT_s100
                                 + 0.882276 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.896891 w_k1_ADM_s100
                                 + 1.189 w_k1_AXP_s100 + 0.912956 w_k1_DIS_s100
                                 + 0.984941 w_k1_FDX_s100
                                 + 1.01649 w_k1_IBM_s100
                                 + 0.81996 w_k1_INTC_s100
                                 + 0.950108 w_k1_KO_s100
                                 + 1.49881 w_k1_MAT_s100
                                 + 0.847408 w_k1_MSFT_s100
                                 + 0.938111 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.2746 w_k1_ADM_s100
                                 + 1.018 w_k1_AXP_s100 + 0.891201 w_k1_DIS_s100
                                 + 1.06839 w_k1_FDX_s100
                                 + 0.803173 w_k1_IBM_s100
                                 + 1.03614 w_k1_INTC_s100
                                 + 0.850735 w_k1_KO_s100
                                 + 0.900588 w_k1_MAT_s100
                                 + 1.02155 w_k1_MSFT_s100
                                 + 1.01019 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.805987 w_k1_ADM_s100
                                 + 0.725368 w_k1_AXP_s100
                                 + 0.548197 w_k1_DIS_s100
                                 + 0.707754 w_k1_FDX_s100
                                 + 0.735709 w_k1_IBM_s100
                                 + 0.860865 w_k1_INTC_s100
                                 + 0.712805 w_k1_KO_s100
                                 + 0.951572 w_k1_MAT_s100
                                 + 0.715303 w_k1_MSFT_s100
                                 + 0.711695 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.7521 w_k1_ADM_s101
                                 + 0.86362 w_k1_AXP_s101
                                 + 1.03015 w_k1_DIS_s101
                                 + 0.883375 w_k1_FDX_s101
                                 + 1.03142 w_k1_IBM_s101
                                 + 1.01684 w_k1_INTC_s101
                                 + 0.931531 w_k1_KO_s101
                                 + 0.768888 w_k1_MAT_s101
                                 + 0.990861 w_k1_MSFT_s101
                                 + 0.906249 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.35883 w_k1_ADM_s101
                                 + 1.74168 w_k1_AXP_s101
                                 + 1.22016 w_k1_DIS_s101
                                 + 1.27104 w_k1_FDX_s101
                                 + 1.04869 w_k1_IBM_s101
                                 + 1.08943 w_k1_INTC_s101
                                 + 0.934011 w_k1_KO_s101
                                 + 1.62166 w_k1_MAT_s101
                                 + 1.07333 w_k1_MSFT_s101 + 1.0506 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.00595 w_k1_ADM_s101
                                 + 1.24928 w_k1_AXP_s101
                                 + 0.922815 w_k1_DIS_s101
                                 + 1.06095 w_k1_FDX_s101
                                 + 1.17564 w_k1_IBM_s101
                                 + 1.08023 w_k1_INTC_s101
                                 + 1.05452 w_k1_KO_s101 + 1.24596 w_k1_MAT_s101
                                 + 1.07021 w_k1_MSFT_s101
                                 + 1.07339 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.24585 w_k1_ADM_s101
                                 + 1.35993 w_k1_AXP_s101
                                 + 1.62627 w_k1_DIS_s101
                                 + 1.26615 w_k1_FDX_s101 + 1.4477 w_k1_IBM_s101
                                 + 0.915672 w_k1_INTC_s101
                                 + 1.25219 w_k1_KO_s101 + 1.43408 w_k1_MAT_s101
                                 + 1.16511 w_k1_MSFT_s101
                                 + 1.28478 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.949489 w_k1_ADM_s101
                                 + 1.07915 w_k1_AXP_s101
                                 + 0.961783 w_k1_DIS_s101
                                 + 1.0914 w_k1_FDX_s101
                                 + 0.795924 w_k1_IBM_s101
                                 + 1.04461 w_k1_INTC_s101
                                 + 0.925121 w_k1_KO_s101
                                 + 0.82054 w_k1_MAT_s101
                                 + 0.998936 w_k1_MSFT_s101
                                 + 1.01169 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.880076 w_k1_ADM_s101
                                 + 0.756606 w_k1_AXP_s101
                                 + 0.770597 w_k1_DIS_s101
                                 + 0.812436 w_k1_FDX_s101
                                 + 1.15446 w_k1_IBM_s101
                                 + 0.819249 w_k1_INTC_s101
                                 + 0.969732 w_k1_KO_s101
                                 + 1.14624 w_k1_MAT_s101
                                 + 0.913934 w_k1_MSFT_s101
                                 + 0.984835 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.00587 w_k1_ADM_s101
                                 + 0.883577 w_k1_AXP_s101
                                 + 1.2817 w_k1_DIS_s101 + 1.14719 w_k1_FDX_s101
                                 + 0.982704 w_k1_IBM_s101
                                 + 1.18215 w_k1_INTC_s101
                                 + 1.19514 w_k1_KO_s101
                                 + 0.868481 w_k1_MAT_s101
                                 + 1.36803 w_k1_MSFT_s101
                                 + 1.10592 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.780722 w_k1_ADM_s101
                                 + 0.819883 w_k1_AXP_s101
                                 + 0.671608 w_k1_DIS_s101
                                 + 0.805984 w_k1_FDX_s101
                                 + 0.72955 w_k1_IBM_s101
                                 + 0.893158 w_k1_INTC_s101
                                 + 0.666135 w_k1_KO_s101
                                 + 0.697837 w_k1_MAT_s101
                                 + 0.82245 w_k1_MSFT_s101
                                 + 0.813431 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.29391 w_k1_ADM_s102
                                 + 0.72545 w_k1_AXP_s102
                                 + 0.713963 w_k1_DIS_s102
                                 + 0.84226 w_k1_FDX_s102
                                 + 0.609841 w_k1_IBM_s102
                                 + 0.722348 w_k1_INTC_s102
                                 + 0.901639 w_k1_KO_s102
                                 + 0.666515 w_k1_MAT_s102
                                 + 0.896672 w_k1_MSFT_s102
                                 + 0.949079 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.45342 w_k1_ADM_s102
                                 + 1.17731 w_k1_AXP_s102
                                 + 1.32656 w_k1_DIS_s102
                                 + 0.974221 w_k1_FDX_s102
                                 + 1.71357 w_k1_IBM_s102
                                 + 0.833859 w_k1_INTC_s102
                                 + 1.40099 w_k1_KO_s102 + 1.83522 w_k1_MAT_s102
                                 + 1.03859 w_k1_MSFT_s102
                                 + 1.08652 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.33274 w_k1_ADM_s102
                                 + 0.915516 w_k1_AXP_s102
                                 + 1.32526 w_k1_DIS_s102
                                 + 1.13752 w_k1_FDX_s102
                                 + 0.830298 w_k1_IBM_s102
                                 + 0.851592 w_k1_INTC_s102
                                 + 1.14469 w_k1_KO_s102
                                 + 0.797942 w_k1_MAT_s102
                                 + 1.224 w_k1_MSFT_s102 + 1.13254 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.15901 w_k1_ADM_s102
                                 + 1.10904 w_k1_AXP_s102
                                 + 1.14869 w_k1_DIS_s102
                                 + 1.03112 w_k1_FDX_s102
                                 + 1.31074 w_k1_IBM_s102
                                 + 1.03019 w_k1_INTC_s102
                                 + 1.22732 w_k1_KO_s102 + 1.19221 w_k1_MAT_s102
                                 + 1.22105 w_k1_MSFT_s102
                                 + 1.16371 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.18699 w_k1_ADM_s102
                                 + 0.909754 w_k1_AXP_s102
                                 + 0.814768 w_k1_DIS_s102
                                 + 0.848065 w_k1_FDX_s102
                                 + 0.932849 w_k1_IBM_s102
                                 + 0.753191 w_k1_INTC_s102
                                 + 0.983469 w_k1_KO_s102
                                 + 0.890923 w_k1_MAT_s102
                                 + 0.847672 w_k1_MSFT_s102
                                 + 0.937341 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.25129 w_k1_ADM_s102
                                 + 0.937375 w_k1_AXP_s102
                                 + 1.03723 w_k1_DIS_s102
                                 + 1.11043 w_k1_FDX_s102
                                 + 0.663446 w_k1_IBM_s102
                                 + 0.852133 w_k1_INTC_s102
                                 + 0.802884 w_k1_KO_s102
                                 + 0.564694 w_k1_MAT_s102
                                 + 0.981541 w_k1_MSFT_s102
                                 + 0.97416 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.23429 w_k1_ADM_s102
                                 + 1.20366 w_k1_AXP_s102
                                 + 1.17203 w_k1_DIS_s102
                                 + 1.12373 w_k1_FDX_s102
                                 + 0.781014 w_k1_IBM_s102
                                 + 1.02947 w_k1_INTC_s102
                                 + 1.04956 w_k1_KO_s102
                                 + 0.900691 w_k1_MAT_s102
                                 + 1.15884 w_k1_MSFT_s102
                                 + 1.01452 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.26895 w_k1_ADM_s102
                                 + 1.38224 w_k1_AXP_s102
                                 + 1.36613 w_k1_DIS_s102
                                 + 1.09535 w_k1_FDX_s102
                                 + 1.06513 w_k1_IBM_s102
                                 + 1.13999 w_k1_INTC_s102
                                 + 1.27502 w_k1_KO_s102 + 1.14296 w_k1_MAT_s102
                                 + 1.24805 w_k1_MSFT_s102
                                 + 1.12099 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.02084 w_k1_ADM_s103
                                 + 1.20666 w_k1_AXP_s103
                                 + 1.00433 w_k1_DIS_s103
                                 + 0.88008 w_k1_FDX_s103
                                 + 1.38835 w_k1_IBM_s103
                                 + 0.94906 w_k1_INTC_s103 + 1.1474 w_k1_KO_s103
                                 + 1.23 w_k1_MAT_s103 + 0.946597 w_k1_MSFT_s103
                                 + 1.03566 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.05086 w_k1_ADM_s103
                                 + 0.851982 w_k1_AXP_s103
                                 + 1.22537 w_k1_DIS_s103 + 0.8887 w_k1_FDX_s103
                                 + 1.32707 w_k1_IBM_s103
                                 + 0.989785 w_k1_INTC_s103
                                 + 1.45674 w_k1_KO_s103 + 1.01511 w_k1_MAT_s103
                                 + 1.23062 w_k1_MSFT_s103
                                 + 0.996485 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.3492 w_k1_ADM_s103
                                 + 1.13087 w_k1_AXP_s103
                                 + 1.17047 w_k1_DIS_s103
                                 + 1.01982 w_k1_FDX_s103
                                 + 1.13726 w_k1_IBM_s103
                                 + 1.09261 w_k1_INTC_s103
                                 + 1.29997 w_k1_KO_s103
                                 + 0.936723 w_k1_MAT_s103
                                 + 1.14872 w_k1_MSFT_s103
                                 + 1.15573 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.02266 w_k1_ADM_s103
                                 + 0.752862 w_k1_AXP_s103
                                 + 0.637868 w_k1_DIS_s103
                                 + 0.770333 w_k1_FDX_s103
                                 + 0.695546 w_k1_IBM_s103
                                 + 0.670309 w_k1_INTC_s103
                                 + 0.715765 w_k1_KO_s103
                                 + 0.619326 w_k1_MAT_s103
                                 + 0.706607 w_k1_MSFT_s103
                                 + 0.875199 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.880492 w_k1_ADM_s103
                                 + 0.68301 w_k1_AXP_s103
                                 + 0.761003 w_k1_DIS_s103
                                 + 0.748021 w_k1_FDX_s103
                                 + 1.03212 w_k1_IBM_s103
                                 + 0.768611 w_k1_INTC_s103
                                 + 0.9272 w_k1_KO_s103 + 0.872356 w_k1_MAT_s103
                                 + 0.771817 w_k1_MSFT_s103
                                 + 0.920858 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.35022 w_k1_ADM_s103
                                 + 0.992115 w_k1_AXP_s103
                                 + 0.918912 w_k1_DIS_s103
                                 + 0.998629 w_k1_FDX_s103
                                 + 0.819566 w_k1_IBM_s103
                                 + 0.964335 w_k1_INTC_s103
                                 + 0.863053 w_k1_KO_s103
                                 + 0.828839 w_k1_MAT_s103
                                 + 0.949461 w_k1_MSFT_s103
                                 + 0.983019 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.7758 w_k1_ADM_s103
                                 + 1.60183 w_k1_AXP_s103
                                 + 1.51071 w_k1_DIS_s103
                                 + 1.26294 w_k1_FDX_s103
                                 + 1.17283 w_k1_IBM_s103
                                 + 1.2938 w_k1_INTC_s103 + 1.29661 w_k1_KO_s103
                                 + 1.18814 w_k1_MAT_s103
                                 + 1.20687 w_k1_MSFT_s103
                                 + 1.12759 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.05993 w_k1_ADM_s103
                                 + 1.04682 w_k1_AXP_s103
                                 + 0.746728 w_k1_DIS_s103
                                 + 0.875364 w_k1_FDX_s103
                                 + 0.930035 w_k1_IBM_s103
                                 + 0.811316 w_k1_INTC_s103
                                 + 0.955126 w_k1_KO_s103
                                 + 1.14451 w_k1_MAT_s103
                                 + 0.799737 w_k1_MSFT_s103
                                 + 0.899159 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.33023 w_k1_ADM_s104
                                 + 1.39351 w_k1_AXP_s104
                                 + 1.75608 w_k1_DIS_s104
                                 + 1.25941 w_k1_FDX_s104
                                 + 1.68754 w_k1_IBM_s104
                                 + 1.0397 w_k1_INTC_s104 + 1.60733 w_k1_KO_s104
                                 + 1.41765 w_k1_MAT_s104
                                 + 1.24639 w_k1_MSFT_s104 + 1.3769 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.17374 w_k1_ADM_s104
                                 + 1.39866 w_k1_AXP_s104
                                 + 1.09605 w_k1_DIS_s104
                                 + 1.13676 w_k1_FDX_s104
                                 + 0.849489 w_k1_IBM_s104
                                 + 0.837407 w_k1_INTC_s104
                                 + 0.966177 w_k1_KO_s104
                                 + 1.19323 w_k1_MAT_s104
                                 + 1.08072 w_k1_MSFT_s104
                                 + 0.989086 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.634948 w_k1_ADM_s104
                                 + 0.852168 w_k1_AXP_s104
                                 + 0.628313 w_k1_DIS_s104
                                 + 0.76741 w_k1_FDX_s104
                                 + 0.753202 w_k1_IBM_s104
                                 + 0.600198 w_k1_INTC_s104
                                 + 0.77611 w_k1_KO_s104
                                 + 0.832929 w_k1_MAT_s104
                                 + 0.752053 w_k1_MSFT_s104
                                 + 0.841387 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.795881 w_k1_ADM_s104
                                 + 0.828726 w_k1_AXP_s104
                                 + 0.952817 w_k1_DIS_s104
                                 + 0.889903 w_k1_FDX_s104
                                 + 1.05925 w_k1_IBM_s104
                                 + 0.944089 w_k1_INTC_s104
                                 + 1.02676 w_k1_KO_s104 + 1.1375 w_k1_MAT_s104
                                 + 0.892448 w_k1_MSFT_s104
                                 + 0.933716 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.950855 w_k1_ADM_s104
                                 + 1.2046 w_k1_AXP_s104 + 1.11124 w_k1_DIS_s104
                                 + 1.09344 w_k1_FDX_s104
                                 + 1.03932 w_k1_IBM_s104
                                 + 0.992254 w_k1_INTC_s104
                                 + 1.08852 w_k1_KO_s104 + 1.19083 w_k1_MAT_s104
                                 + 1.02047 w_k1_MSFT_s104
                                 + 1.09236 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.06363 w_k1_ADM_s104
                                 + 1.09503 w_k1_AXP_s104
                                 + 1.47164 w_k1_DIS_s104
                                 + 1.16244 w_k1_FDX_s104
                                 + 1.28212 w_k1_IBM_s104
                                 + 1.07224 w_k1_INTC_s104
                                 + 1.54852 w_k1_KO_s104 + 1.33166 w_k1_MAT_s104
                                 + 1.19707 w_k1_MSFT_s104
                                 + 1.25571 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.727393 w_k1_ADM_s104
                                 + 0.923838 w_k1_AXP_s104
                                 + 1.1289 w_k1_DIS_s104 + 0.92221 w_k1_FDX_s104
                                 + 1.41858 w_k1_IBM_s104
                                 + 0.957154 w_k1_INTC_s104
                                 + 1.14835 w_k1_KO_s104 + 1.58264 w_k1_MAT_s104
                                 + 0.934471 w_k1_MSFT_s104
                                 + 1.03515 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.815563 w_k1_ADM_s104
                                 + 0.816898 w_k1_AXP_s104
                                 + 0.842045 w_k1_DIS_s104
                                 + 0.933674 w_k1_FDX_s104
                                 + 0.694229 w_k1_IBM_s104
                                 + 0.883374 w_k1_INTC_s104
                                 + 0.920045 w_k1_KO_s104
                                 + 0.726136 w_k1_MAT_s104
                                 + 0.870911 w_k1_MSFT_s104
                                 + 0.956389 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.966247 w_k1_ADM_s105
                                 + 0.995978 w_k1_AXP_s105
                                 + 0.944519 w_k1_DIS_s105
                                 + 0.98743 w_k1_FDX_s105
                                 + 1.59483 w_k1_IBM_s105
                                 + 0.934025 w_k1_INTC_s105
                                 + 1.32659 w_k1_KO_s105 + 2.05057 w_k1_MAT_s105
                                 + 1.00662 w_k1_MSFT_s105
                                 + 1.10709 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.03491 w_k1_ADM_s105
                                 + 1.16305 w_k1_AXP_s105
                                 + 1.27592 w_k1_DIS_s105
                                 + 1.14451 w_k1_FDX_s105
                                 + 1.28121 w_k1_IBM_s105
                                 + 1.11673 w_k1_INTC_s105
                                 + 1.20817 w_k1_KO_s105 + 1.14327 w_k1_MAT_s105
                                 + 1.05198 w_k1_MSFT_s105 + 1.0517 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.31012 w_k1_ADM_s105
                                 + 1.49163 w_k1_AXP_s105
                                 + 1.62032 w_k1_DIS_s105 + 1.4126 w_k1_FDX_s105
                                 + 1.24809 w_k1_IBM_s105
                                 + 1.23226 w_k1_INTC_s105
                                 + 1.57747 w_k1_KO_s105 + 1.86634 w_k1_MAT_s105
                                 + 1.16978 w_k1_MSFT_s105
                                 + 1.38582 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.15053 w_k1_ADM_s105
                                 + 1.3502 w_k1_AXP_s105 + 1.25811 w_k1_DIS_s105
                                 + 1.29343 w_k1_FDX_s105
                                 + 0.900807 w_k1_IBM_s105
                                 + 1.32523 w_k1_INTC_s105
                                 + 1.08272 w_k1_KO_s105 + 1.06741 w_k1_MAT_s105
                                 + 1.14717 w_k1_MSFT_s105
                                 + 1.05001 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.29523 w_k1_ADM_s105
                                 + 0.959672 w_k1_AXP_s105
                                 + 1.03566 w_k1_DIS_s105
                                 + 1.11772 w_k1_FDX_s105
                                 + 0.711808 w_k1_IBM_s105
                                 + 1.03319 w_k1_INTC_s105
                                 + 0.988185 w_k1_KO_s105
                                 + 0.625755 w_k1_MAT_s105
                                 + 1.0682 w_k1_MSFT_s105
                                 + 0.971959 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.833762 w_k1_ADM_s105
                                 + 0.769917 w_k1_AXP_s105
                                 + 0.811316 w_k1_DIS_s105
                                 + 0.871747 w_k1_FDX_s105
                                 + 0.685938 w_k1_IBM_s105
                                 + 0.941889 w_k1_INTC_s105
                                 + 0.83139 w_k1_KO_s105
                                 + 0.679368 w_k1_MAT_s105
                                 + 0.842018 w_k1_MSFT_s105
                                 + 0.807186 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.92737 w_k1_ADM_s105
                                 + 0.735686 w_k1_AXP_s105
                                 + 0.633051 w_k1_DIS_s105
                                 + 0.72131 w_k1_FDX_s105
                                 + 0.795266 w_k1_IBM_s105
                                 + 0.928199 w_k1_INTC_s105
                                 + 0.771717 w_k1_KO_s105
                                 + 0.737324 w_k1_MAT_s105
                                 + 0.719231 w_k1_MSFT_s105
                                 + 0.842273 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.866294 w_k1_ADM_s105
                                 + 0.958852 w_k1_AXP_s105
                                 + 0.92269 w_k1_DIS_s105
                                 + 0.96741 w_k1_FDX_s105 + 1.0478 w_k1_IBM_s105
                                 + 1.08983 w_k1_INTC_s105
                                 + 1.04258 w_k1_KO_s105 + 1.10964 w_k1_MAT_s105
                                 + 0.875202 w_k1_MSFT_s105
                                 + 0.95198 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.832517 w_k1_ADM_s106
                                 + 0.905815 w_k1_AXP_s106
                                 + 1.13291 w_k1_DIS_s106
                                 + 1.05132 w_k1_FDX_s106
                                 + 0.848638 w_k1_IBM_s106
                                 + 1.07938 w_k1_INTC_s106
                                 + 1.09002 w_k1_KO_s106 + 1.10496 w_k1_MAT_s106
                                 + 1.18277 w_k1_MSFT_s106
                                 + 1.06177 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.55522 w_k1_ADM_s106
                                 + 1.71183 w_k1_AXP_s106
                                 + 1.87171 w_k1_DIS_s106
                                 + 1.44923 w_k1_FDX_s106
                                 + 1.59664 w_k1_IBM_s106
                                 + 1.62818 w_k1_INTC_s106
                                 + 1.69398 w_k1_KO_s106 + 1.26943 w_k1_MAT_s106
                                 + 1.3975 w_k1_MSFT_s106 + 1.26897 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.882128 w_k1_ADM_s106
                                 + 1.18789 w_k1_AXP_s106 + 1.0043 w_k1_DIS_s106
                                 + 0.972057 w_k1_FDX_s106
                                 + 1.18234 w_k1_IBM_s106
                                 + 0.916722 w_k1_INTC_s106
                                 + 1.04318 w_k1_KO_s106 + 1.34436 w_k1_MAT_s106
                                 + 0.997542 w_k1_MSFT_s106
                                 + 1.05583 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.846582 w_k1_ADM_s106
                                 + 0.937281 w_k1_AXP_s106
                                 + 1.11372 w_k1_DIS_s106
                                 + 0.896447 w_k1_FDX_s106
                                 + 1.50159 w_k1_IBM_s106
                                 + 1.07117 w_k1_INTC_s106
                                 + 1.28682 w_k1_KO_s106 + 1.3384 w_k1_MAT_s106
                                 + 1.07724 w_k1_MSFT_s106
                                 + 1.03711 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.30008 w_k1_ADM_s106
                                 + 1.28538 w_k1_AXP_s106
                                 + 1.57827 w_k1_DIS_s106
                                 + 1.29819 w_k1_FDX_s106
                                 + 1.14678 w_k1_IBM_s106
                                 + 1.18178 w_k1_INTC_s106
                                 + 1.32425 w_k1_KO_s106 + 1.30841 w_k1_MAT_s106
                                 + 1.31334 w_k1_MSFT_s106
                                 + 1.16712 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.731472 w_k1_ADM_s106
                                 + 0.720461 w_k1_AXP_s106
                                 + 0.815221 w_k1_DIS_s106
                                 + 0.793383 w_k1_FDX_s106
                                 + 0.937128 w_k1_IBM_s106
                                 + 0.775669 w_k1_INTC_s106
                                 + 0.973426 w_k1_KO_s106
                                 + 0.970783 w_k1_MAT_s106
                                 + 0.834402 w_k1_MSFT_s106
                                 + 1.02031 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.0283 w_k1_ADM_s106
                                 + 0.936048 w_k1_AXP_s106
                                 + 0.674379 w_k1_DIS_s106
                                 + 0.85248 w_k1_FDX_s106
                                 + 0.729918 w_k1_IBM_s106
                                 + 0.823903 w_k1_INTC_s106
                                 + 0.800156 w_k1_KO_s106
                                 + 0.784602 w_k1_MAT_s106
                                 + 0.807602 w_k1_MSFT_s106
                                 + 0.922447 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.02055 w_k1_ADM_s106
                                 + 1.39734 w_k1_AXP_s106
                                 + 1.16133 w_k1_DIS_s106
                                 + 1.23919 w_k1_FDX_s106
                                 + 0.948257 w_k1_IBM_s106
                                 + 1.53659 w_k1_INTC_s106
                                 + 1.16364 w_k1_KO_s106 + 1.33207 w_k1_MAT_s106
                                 + 1.18306 w_k1_MSFT_s106 + 1.1406 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.874296 w_k1_ADM_s107
                                 + 1.69408 w_k1_AXP_s107
                                 + 1.30308 w_k1_DIS_s107
                                 + 1.17394 w_k1_FDX_s107
                                 + 1.39624 w_k1_IBM_s107
                                 + 1.27723 w_k1_INTC_s107
                                 + 1.18233 w_k1_KO_s107 + 1.48168 w_k1_MAT_s107
                                 + 0.929926 w_k1_MSFT_s107
                                 + 1.21274 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.08796 w_k1_ADM_s107
                                 + 1.16776 w_k1_AXP_s107
                                 + 1.27434 w_k1_DIS_s107
                                 + 1.20079 w_k1_FDX_s107
                                 + 0.878608 w_k1_IBM_s107
                                 + 1.10953 w_k1_INTC_s107
                                 + 0.927573 w_k1_KO_s107
                                 + 0.827173 w_k1_MAT_s107
                                 + 1.10469 w_k1_MSFT_s107
                                 + 1.13117 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.4132 w_k1_ADM_s107
                                 + 1.90189 w_k1_AXP_s107
                                 + 1.87009 w_k1_DIS_s107
                                 + 1.36084 w_k1_FDX_s107
                                 + 1.19371 w_k1_IBM_s107
                                 + 1.20576 w_k1_INTC_s107
                                 + 1.21619 w_k1_KO_s107 + 1.74073 w_k1_MAT_s107
                                 + 1.41357 w_k1_MSFT_s107
                                 + 1.41899 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.06407 w_k1_ADM_s107
                                 + 1.19096 w_k1_AXP_s107
                                 + 0.854742 w_k1_DIS_s107
                                 + 1.00317 w_k1_FDX_s107
                                 + 1.03495 w_k1_IBM_s107
                                 + 0.94256 w_k1_INTC_s107
                                 + 0.795988 w_k1_KO_s107
                                 + 1.27769 w_k1_MAT_s107
                                 + 0.763676 w_k1_MSFT_s107
                                 + 1.01368 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.81856 w_k1_ADM_s107
                                 + 0.726169 w_k1_AXP_s107
                                 + 0.683397 w_k1_DIS_s107
                                 + 0.742598 w_k1_FDX_s107
                                 + 0.662042 w_k1_IBM_s107
                                 + 0.84169 w_k1_INTC_s107
                                 + 0.664791 w_k1_KO_s107
                                 + 0.687478 w_k1_MAT_s107
                                 + 0.717683 w_k1_MSFT_s107
                                 + 0.878732 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.679388 w_k1_ADM_s107
                                 + 0.803061 w_k1_AXP_s107
                                 + 0.72865 w_k1_DIS_s107
                                 + 0.769697 w_k1_FDX_s107
                                 + 1.01718 w_k1_IBM_s107
                                 + 0.664829 w_k1_INTC_s107
                                 + 0.833698 w_k1_KO_s107 + 1.1537 w_k1_MAT_s107
                                 + 0.674774 w_k1_MSFT_s107
                                 + 0.911217 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.962464 w_k1_ADM_s107
                                 + 0.90797 w_k1_AXP_s107
                                 + 1.04128 w_k1_DIS_s107 + 1.0272 w_k1_FDX_s107
                                 + 0.7578 w_k1_IBM_s107
                                 + 1.15613 w_k1_INTC_s107
                                 + 0.869955 w_k1_KO_s107
                                 + 0.540617 w_k1_MAT_s107
                                 + 0.940562 w_k1_MSFT_s107
                                 + 0.956102 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.804793 w_k1_ADM_s107
                                 + 0.961286 w_k1_AXP_s107
                                 + 0.875159 w_k1_DIS_s107
                                 + 0.853792 w_k1_FDX_s107
                                 + 1.4829 w_k1_IBM_s107
                                 + 1.01643 w_k1_INTC_s107
                                 + 1.04727 w_k1_KO_s107 + 1.26882 w_k1_MAT_s107
                                 + 0.812579 w_k1_MSFT_s107
                                 + 0.923431 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.03105 w_k1_ADM_s110
                                 + 0.917196 w_k1_AXP_s110
                                 + 0.947324 w_k1_DIS_s110
                                 + 1.01816 w_k1_FDX_s110
                                 + 0.951264 w_k1_IBM_s110
                                 + 0.803011 w_k1_INTC_s110
                                 + 0.864545 w_k1_KO_s110
                                 + 0.908646 w_k1_MAT_s110
                                 + 0.985887 w_k1_MSFT_s110
                                 + 0.95222 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.04936 w_k1_ADM_s110
                                 + 1.05263 w_k1_AXP_s110
                                 + 1.42322 w_k1_DIS_s110
                                 + 1.05559 w_k1_FDX_s110
                                 + 1.27642 w_k1_IBM_s110
                                 + 1.03034 w_k1_INTC_s110
                                 + 1.17838 w_k1_KO_s110 + 1.30951 w_k1_MAT_s110
                                 + 1.01838 w_k1_MSFT_s110
                                 + 1.15402 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09824 w_k1_ADM_s110
                                 + 0.683918 w_k1_AXP_s110
                                 + 0.692255 w_k1_DIS_s110
                                 + 0.752367 w_k1_FDX_s110
                                 + 0.937312 w_k1_IBM_s110
                                 + 0.73433 w_k1_INTC_s110
                                 + 0.782853 w_k1_KO_s110
                                 + 0.950016 w_k1_MAT_s110
                                 + 0.722118 w_k1_MSFT_s110
                                 + 0.912308 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.896925 w_k1_ADM_s110
                                 + 0.547712 w_k1_AXP_s110
                                 + 0.665624 w_k1_DIS_s110
                                 + 0.798298 w_k1_FDX_s110
                                 + 0.584262 w_k1_IBM_s110
                                 + 0.779936 w_k1_INTC_s110
                                 + 0.622878 w_k1_KO_s110
                                 + 0.435941 w_k1_MAT_s110
                                 + 0.818116 w_k1_MSFT_s110
                                 + 0.783696 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.58344 w_k1_ADM_s110
                                 + 1.58332 w_k1_AXP_s110
                                 + 1.76768 w_k1_DIS_s110
                                 + 1.36524 w_k1_FDX_s110
                                 + 1.54306 w_k1_IBM_s110
                                 + 1.31019 w_k1_INTC_s110
                                 + 1.52318 w_k1_KO_s110 + 1.92136 w_k1_MAT_s110
                                 + 1.33763 w_k1_MSFT_s110
                                 + 1.42198 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.41643 w_k1_ADM_s110
                                 + 1.0852 w_k1_AXP_s110 + 1.34811 w_k1_DIS_s110
                                 + 1.30386 w_k1_FDX_s110
                                 + 0.861692 w_k1_IBM_s110
                                 + 0.867579 w_k1_INTC_s110
                                 + 1.06828 w_k1_KO_s110
                                 + 0.956461 w_k1_MAT_s110
                                 + 1.20995 w_k1_MSFT_s110
                                 + 1.11695 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.26088 w_k1_ADM_s110
                                 + 1.16034 w_k1_AXP_s110
                                 + 1.21353 w_k1_DIS_s110
                                 + 1.21162 w_k1_FDX_s110
                                 + 1.08788 w_k1_IBM_s110
                                 + 0.875074 w_k1_INTC_s110
                                 + 1.07444 w_k1_KO_s110 + 1.02805 w_k1_MAT_s110
                                 + 0.949609 w_k1_MSFT_s110
                                 + 1.01454 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.06625 w_k1_ADM_s110
                                 + 0.914206 w_k1_AXP_s110
                                 + 0.961045 w_k1_DIS_s110
                                 + 0.904625 w_k1_FDX_s110
                                 + 1.4202 w_k1_IBM_s110
                                 + 0.992142 w_k1_INTC_s110
                                 + 1.04458 w_k1_KO_s110 + 1.38528 w_k1_MAT_s110
                                 + 0.928178 w_k1_MSFT_s110
                                 + 0.963913 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.07676 w_k1_ADM_s111
                                 + 0.840658 w_k1_AXP_s111
                                 + 0.757519 w_k1_DIS_s111
                                 + 0.870561 w_k1_FDX_s111
                                 + 0.712915 w_k1_IBM_s111
                                 + 0.662361 w_k1_INTC_s111
                                 + 0.773071 w_k1_KO_s111
                                 + 0.598052 w_k1_MAT_s111
                                 + 0.853894 w_k1_MSFT_s111
                                 + 0.725945 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.976821 w_k1_ADM_s111
                                 + 0.915046 w_k1_AXP_s111
                                 + 0.858233 w_k1_DIS_s111
                                 + 0.995048 w_k1_FDX_s111
                                 + 0.694637 w_k1_IBM_s111
                                 + 0.824084 w_k1_INTC_s111
                                 + 0.870665 w_k1_KO_s111
                                 + 0.722614 w_k1_MAT_s111
                                 + 0.964798 w_k1_MSFT_s111
                                 + 0.934404 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.842188 w_k1_ADM_s111
                                 + 0.78941 w_k1_AXP_s111
                                 + 0.786129 w_k1_DIS_s111
                                 + 0.789893 w_k1_FDX_s111
                                 + 1.09105 w_k1_IBM_s111
                                 + 0.747952 w_k1_INTC_s111
                                 + 0.911498 w_k1_KO_s111
                                 + 0.990958 w_k1_MAT_s111
                                 + 0.704356 w_k1_MSFT_s111
                                 + 0.824394 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.990622 w_k1_ADM_s111
                                 + 1.38034 w_k1_AXP_s111
                                 + 1.41796 w_k1_DIS_s111
                                 + 1.19769 w_k1_FDX_s111
                                 + 1.43581 w_k1_IBM_s111
                                 + 1.16077 w_k1_INTC_s111
                                 + 1.23648 w_k1_KO_s111 + 1.29343 w_k1_MAT_s111
                                 + 1.13882 w_k1_MSFT_s111
                                 + 0.992211 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.13879 w_k1_ADM_s111
                                 + 1.1533 w_k1_AXP_s111 + 1.34824 w_k1_DIS_s111
                                 + 1.24156 w_k1_FDX_s111
                                 + 0.868759 w_k1_IBM_s111
                                 + 1.07913 w_k1_INTC_s111
                                 + 1.05987 w_k1_KO_s111
                                 + 0.890702 w_k1_MAT_s111
                                 + 1.13679 w_k1_MSFT_s111
                                 + 1.00297 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.731651 w_k1_ADM_s111
                                 + 0.762088 w_k1_AXP_s111
                                 + 0.790044 w_k1_DIS_s111
                                 + 0.738819 w_k1_FDX_s111
                                 + 0.947787 w_k1_IBM_s111
                                 + 0.742017 w_k1_INTC_s111
                                 + 0.850385 w_k1_KO_s111
                                 + 0.562587 w_k1_MAT_s111
                                 + 0.77087 w_k1_MSFT_s111
                                 + 0.836564 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.18462 w_k1_ADM_s111
                                 + 0.946242 w_k1_AXP_s111
                                 + 1.01641 w_k1_DIS_s111
                                 + 1.04684 w_k1_FDX_s111
                                 + 0.954839 w_k1_IBM_s111
                                 + 0.901621 w_k1_INTC_s111
                                 + 0.941207 w_k1_KO_s111
                                 + 0.737828 w_k1_MAT_s111
                                 + 1.00761 w_k1_MSFT_s111
                                 + 0.952606 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.887805 w_k1_ADM_s111
                                 + 1.22504 w_k1_AXP_s111
                                 + 1.02828 w_k1_DIS_s111
                                 + 1.06395 w_k1_FDX_s111
                                 + 1.01152 w_k1_IBM_s111
                                 + 0.984115 w_k1_INTC_s111
                                 + 0.994999 w_k1_KO_s111 + 1.0849 w_k1_MAT_s111
                                 + 1.06313 w_k1_MSFT_s111
                                 + 1.02129 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.58086 w_k1_ADM_s112
                                 + 1.2588 w_k1_AXP_s112 + 1.20518 w_k1_DIS_s112
                                 + 1.2959 w_k1_FDX_s112
                                 + 0.714935 w_k1_IBM_s112
                                 + 1.16357 w_k1_INTC_s112
                                 + 0.844761 w_k1_KO_s112
                                 + 0.563599 w_k1_MAT_s112
                                 + 1.19987 w_k1_MSFT_s112
                                 + 0.914077 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.814843 w_k1_ADM_s112
                                 + 0.828621 w_k1_AXP_s112
                                 + 0.618359 w_k1_DIS_s112
                                 + 0.744983 w_k1_FDX_s112
                                 + 0.82161 w_k1_IBM_s112
                                 + 0.874434 w_k1_INTC_s112
                                 + 0.816768 w_k1_KO_s112 + 0.8604 w_k1_MAT_s112
                                 + 0.877033 w_k1_MSFT_s112
                                 + 0.773185 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.19032 w_k1_ADM_s112
                                 + 1.36015 w_k1_AXP_s112
                                 + 1.01651 w_k1_DIS_s112
                                 + 1.05766 w_k1_FDX_s112
                                 + 0.838831 w_k1_IBM_s112
                                 + 0.899309 w_k1_INTC_s112
                                 + 0.879389 w_k1_KO_s112
                                 + 1.01354 w_k1_MAT_s112
                                 + 1.07685 w_k1_MSFT_s112
                                 + 0.89953 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.05133 w_k1_ADM_s112
                                 + 0.992903 w_k1_AXP_s112
                                 + 1.35664 w_k1_DIS_s112
                                 + 1.02309 w_k1_FDX_s112
                                 + 1.18789 w_k1_IBM_s112
                                 + 1.02095 w_k1_INTC_s112 + 1.2333 w_k1_KO_s112
                                 + 1.26456 w_k1_MAT_s112
                                 + 1.33658 w_k1_MSFT_s112
                                 + 1.06641 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.940387 w_k1_ADM_s112
                                 + 0.86444 w_k1_AXP_s112
                                 + 0.892734 w_k1_DIS_s112
                                 + 0.927098 w_k1_FDX_s112
                                 + 0.874221 w_k1_IBM_s112
                                 + 0.958302 w_k1_INTC_s112
                                 + 0.971299 w_k1_KO_s112
                                 + 0.701574 w_k1_MAT_s112
                                 + 1.04328 w_k1_MSFT_s112
                                 + 0.921118 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.04874 w_k1_ADM_s112
                                 + 1.16776 w_k1_AXP_s112
                                 + 1.03434 w_k1_DIS_s112
                                 + 0.97446 w_k1_FDX_s112
                                 + 1.28276 w_k1_IBM_s112
                                 + 0.827676 w_k1_INTC_s112
                                 + 1.09648 w_k1_KO_s112 + 1.76331 w_k1_MAT_s112
                                 + 1.16755 w_k1_MSFT_s112
                                 + 0.933893 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.00918 w_k1_ADM_s112
                                 + 0.982378 w_k1_AXP_s112
                                 + 0.808936 w_k1_DIS_s112
                                 + 0.849616 w_k1_FDX_s112
                                 + 1.34348 w_k1_IBM_s112
                                 + 0.918622 w_k1_INTC_s112
                                 + 1.10472 w_k1_KO_s112 + 1.67664 w_k1_MAT_s112
                                 + 0.997811 w_k1_MSFT_s112
                                 + 0.853482 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.34495 w_k1_ADM_s112
                                 + 1.35218 w_k1_AXP_s112
                                 + 1.31815 w_k1_DIS_s112
                                 + 1.32902 w_k1_FDX_s112
                                 + 1.18624 w_k1_IBM_s112
                                 + 1.24311 w_k1_INTC_s112
                                 + 1.23287 w_k1_KO_s112 + 1.12676 w_k1_MAT_s112
                                 + 1.38912 w_k1_MSFT_s112 + 1.1945 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.852201 w_k1_ADM_s113
                                 + 0.896722 w_k1_AXP_s113
                                 + 0.87167 w_k1_DIS_s113
                                 + 1.10331 w_k1_FDX_s113
                                 + 0.677883 w_k1_IBM_s113
                                 + 1.11578 w_k1_INTC_s113
                                 + 0.749432 w_k1_KO_s113
                                 + 0.589691 w_k1_MAT_s113
                                 + 0.985079 w_k1_MSFT_s113
                                 + 0.800909 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.18074 w_k1_ADM_s113
                                 + 1.35957 w_k1_AXP_s113
                                 + 1.48944 w_k1_DIS_s113
                                 + 1.37791 w_k1_FDX_s113
                                 + 0.99741 w_k1_IBM_s113
                                 + 1.48243 w_k1_INTC_s113 + 1.2197 w_k1_KO_s113
                                 + 1.25166 w_k1_MAT_s113
                                 + 1.34593 w_k1_MSFT_s113
                                 + 1.12111 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.01227 w_k1_ADM_s113
                                 + 1.0062 w_k1_AXP_s113 + 1.08467 w_k1_DIS_s113
                                 + 1.00368 w_k1_FDX_s113
                                 + 1.56424 w_k1_IBM_s113
                                 + 0.934493 w_k1_INTC_s113
                                 + 1.22167 w_k1_KO_s113 + 1.62222 w_k1_MAT_s113
                                 + 1.05745 w_k1_MSFT_s113
                                 + 1.09146 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.929224 w_k1_ADM_s113
                                 + 1.40033 w_k1_AXP_s113
                                 + 1.40198 w_k1_DIS_s113
                                 + 1.25879 w_k1_FDX_s113
                                 + 1.31499 w_k1_IBM_s113
                                 + 1.26904 w_k1_INTC_s113
                                 + 1.06115 w_k1_KO_s113 + 1.3203 w_k1_MAT_s113
                                 + 1.27846 w_k1_MSFT_s113
                                 + 0.99328 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.921074 w_k1_ADM_s113
                                 + 0.944795 w_k1_AXP_s113
                                 + 0.757864 w_k1_DIS_s113
                                 + 0.929123 w_k1_FDX_s113
                                 + 1.04316 w_k1_IBM_s113
                                 + 0.895492 w_k1_INTC_s113
                                 + 0.826774 w_k1_KO_s113
                                 + 1.39848 w_k1_MAT_s113
                                 + 0.896565 w_k1_MSFT_s113
                                 + 0.90336 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.757812 w_k1_ADM_s113
                                 + 1.05063 w_k1_AXP_s113
                                 + 0.994952 w_k1_DIS_s113
                                 + 1.04841 w_k1_FDX_s113
                                 + 0.799102 w_k1_IBM_s113
                                 + 1.13075 w_k1_INTC_s113
                                 + 0.753625 w_k1_KO_s113
                                 + 0.87584 w_k1_MAT_s113
                                 + 0.941278 w_k1_MSFT_s113
                                 + 0.868815 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.858022 w_k1_ADM_s113
                                 + 0.911099 w_k1_AXP_s113
                                 + 1.04517 w_k1_DIS_s113
                                 + 1.04632 w_k1_FDX_s113
                                 + 1.00704 w_k1_IBM_s113
                                 + 1.01892 w_k1_INTC_s113
                                 + 0.982055 w_k1_KO_s113
                                 + 0.927411 w_k1_MAT_s113
                                 + 1.07483 w_k1_MSFT_s113
                                 + 0.966951 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.88795 w_k1_ADM_s113
                                 + 0.714051 w_k1_AXP_s113
                                 + 0.651909 w_k1_DIS_s113
                                 + 0.829006 w_k1_FDX_s113
                                 + 0.797537 w_k1_IBM_s113
                                 + 0.840059 w_k1_INTC_s113
                                 + 0.733663 w_k1_KO_s113
                                 + 0.846516 w_k1_MAT_s113
                                 + 0.832504 w_k1_MSFT_s113
                                 + 0.811735 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.0431 w_k1_ADM_s114
                                 + 1.08691 w_k1_AXP_s114
                                 + 1.11213 w_k1_DIS_s114
                                 + 1.10309 w_k1_FDX_s114
                                 + 0.914102 w_k1_IBM_s114
                                 + 0.994143 w_k1_INTC_s114
                                 + 0.955617 w_k1_KO_s114
                                 + 1.27872 w_k1_MAT_s114
                                 + 1.0973 w_k1_MSFT_s114 + 1.16105 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.08277 w_k1_ADM_s114
                                 + 0.896358 w_k1_AXP_s114
                                 + 0.945828 w_k1_DIS_s114
                                 + 1.02082 w_k1_FDX_s114
                                 + 0.67561 w_k1_IBM_s114
                                 + 1.09323 w_k1_INTC_s114
                                 + 0.735962 w_k1_KO_s114
                                 + 0.532036 w_k1_MAT_s114
                                 + 0.972881 w_k1_MSFT_s114
                                 + 0.95408 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.05409 w_k1_ADM_s114
                                 + 1.23496 w_k1_AXP_s114
                                 + 1.33326 w_k1_DIS_s114
                                 + 1.13253 w_k1_FDX_s114
                                 + 1.57586 w_k1_IBM_s114
                                 + 1.02341 w_k1_INTC_s114
                                 + 1.24105 w_k1_KO_s114 + 2.01688 w_k1_MAT_s114
                                 + 1.09448 w_k1_MSFT_s114
                                 + 1.23755 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.712533 w_k1_ADM_s114
                                 + 0.725342 w_k1_AXP_s114
                                 + 0.730469 w_k1_DIS_s114
                                 + 0.726115 w_k1_FDX_s114
                                 + 0.87111 w_k1_IBM_s114
                                 + 0.787618 w_k1_INTC_s114
                                 + 0.746074 w_k1_KO_s114
                                 + 0.959655 w_k1_MAT_s114
                                 + 0.762746 w_k1_MSFT_s114
                                 + 0.863291 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.11215 w_k1_ADM_s114
                                 + 1.09419 w_k1_AXP_s114
                                 + 0.807375 w_k1_DIS_s114
                                 + 0.898979 w_k1_FDX_s114
                                 + 0.877702 w_k1_IBM_s114
                                 + 0.913971 w_k1_INTC_s114
                                 + 0.73624 w_k1_KO_s114 + 0.99779 w_k1_MAT_s114
                                 + 0.72899 w_k1_MSFT_s114
                                 + 0.950175 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.987244 w_k1_ADM_s114
                                 + 0.981957 w_k1_AXP_s114 + 1.009 w_k1_DIS_s114
                                 + 0.944647 w_k1_FDX_s114
                                 + 1.18857 w_k1_IBM_s114
                                 + 0.814658 w_k1_INTC_s114
                                 + 0.928135 w_k1_KO_s114
                                 + 1.66739 w_k1_MAT_s114
                                 + 0.937956 w_k1_MSFT_s114
                                 + 1.05902 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.882348 w_k1_ADM_s114
                                 + 0.808753 w_k1_AXP_s114
                                 + 1.02501 w_k1_DIS_s114
                                 + 0.950044 w_k1_FDX_s114
                                 + 0.802641 w_k1_IBM_s114
                                 + 1.0362 w_k1_INTC_s114
                                 + 0.830695 w_k1_KO_s114
                                 + 0.802052 w_k1_MAT_s114
                                 + 0.969327 w_k1_MSFT_s114
                                 + 1.00669 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.12732 w_k1_ADM_s114
                                 + 1.33216 w_k1_AXP_s114
                                 + 1.60305 w_k1_DIS_s114
                                 + 1.29107 w_k1_FDX_s114
                                 + 1.16862 w_k1_IBM_s114
                                 + 1.31913 w_k1_INTC_s114
                                 + 1.14112 w_k1_KO_s114 + 1.11053 w_k1_MAT_s114
                                 + 1.21203 w_k1_MSFT_s114
                                 + 1.25696 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.21628 w_k1_ADM_s115
                                 + 1.30288 w_k1_AXP_s115
                                 + 1.30087 w_k1_DIS_s115
                                 + 1.26345 w_k1_FDX_s115
                                 + 1.20209 w_k1_IBM_s115
                                 + 1.20997 w_k1_INTC_s115
                                 + 1.19489 w_k1_KO_s115 + 1.06459 w_k1_MAT_s115
                                 + 1.15782 w_k1_MSFT_s115
                                 + 1.11072 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11236 w_k1_ADM_s115
                                 + 1.0093 w_k1_AXP_s115
                                 + 0.844589 w_k1_DIS_s115
                                 + 1.05896 w_k1_FDX_s115
                                 + 0.740581 w_k1_IBM_s115
                                 + 0.903086 w_k1_INTC_s115
                                 + 0.767296 w_k1_KO_s115
                                 + 1.05322 w_k1_MAT_s115
                                 + 0.950255 w_k1_MSFT_s115
                                 + 0.923269 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.32258 w_k1_ADM_s115
                                 + 1.25036 w_k1_AXP_s115
                                 + 0.954905 w_k1_DIS_s115
                                 + 1.06294 w_k1_FDX_s115
                                 + 1.38707 w_k1_IBM_s115
                                 + 1.0351 w_k1_INTC_s115 + 1.057 w_k1_KO_s115
                                 + 1.56139 w_k1_MAT_s115
                                 + 0.987219 w_k1_MSFT_s115
                                 + 1.06998 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.809603 w_k1_ADM_s115
                                 + 0.619631 w_k1_AXP_s115
                                 + 0.58481 w_k1_DIS_s115
                                 + 0.743119 w_k1_FDX_s115
                                 + 0.907244 w_k1_IBM_s115
                                 + 0.724415 w_k1_INTC_s115
                                 + 0.912341 w_k1_KO_s115
                                 + 1.08733 w_k1_MAT_s115
                                 + 0.749034 w_k1_MSFT_s115
                                 + 0.893287 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.19212 w_k1_ADM_s115
                                 + 1.25826 w_k1_AXP_s115
                                 + 1.88248 w_k1_DIS_s115
                                 + 1.46543 w_k1_FDX_s115
                                 + 1.42405 w_k1_IBM_s115
                                 + 1.3254 w_k1_INTC_s115 + 1.54795 w_k1_KO_s115
                                 + 1.92677 w_k1_MAT_s115
                                 + 1.43574 w_k1_MSFT_s115
                                 + 1.26842 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.994132 w_k1_ADM_s115
                                 + 0.898677 w_k1_AXP_s115
                                 + 1.07959 w_k1_DIS_s115 + 1.1133 w_k1_FDX_s115
                                 + 0.671079 w_k1_IBM_s115
                                 + 0.980347 w_k1_INTC_s115
                                 + 0.896409 w_k1_KO_s115
                                 + 0.839047 w_k1_MAT_s115
                                 + 1.07732 w_k1_MSFT_s115
                                 + 0.995218 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.00078 w_k1_ADM_s115
                                 + 0.88467 w_k1_AXP_s115
                                 + 1.06097 w_k1_DIS_s115
                                 + 0.929259 w_k1_FDX_s115
                                 + 1.05953 w_k1_IBM_s115
                                 + 1.06421 w_k1_INTC_s115
                                 + 1.32795 w_k1_KO_s115 + 1.26563 w_k1_MAT_s115
                                 + 1.10806 w_k1_MSFT_s115
                                 + 1.06841 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.15142 w_k1_ADM_s115
                                 + 1.13366 w_k1_AXP_s115
                                 + 0.824702 w_k1_DIS_s115
                                 + 0.935929 w_k1_FDX_s115
                                 + 1.17909 w_k1_IBM_s115
                                 + 0.891451 w_k1_INTC_s115
                                 + 1.04783 w_k1_KO_s115 + 1.15593 w_k1_MAT_s115
                                 + 0.881189 w_k1_MSFT_s115
                                 + 1.01913 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.04676 w_k1_ADM_s116
                                 + 1.12339 w_k1_AXP_s116
                                 + 1.24803 w_k1_DIS_s116
                                 + 1.10287 w_k1_FDX_s116
                                 + 1.26494 w_k1_IBM_s116
                                 + 1.17784 w_k1_INTC_s116
                                 + 1.18003 w_k1_KO_s116 + 1.19459 w_k1_MAT_s116
                                 + 1.14988 w_k1_MSFT_s116 + 1.2706 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.29464 w_k1_ADM_s116
                                 + 1.68088 w_k1_AXP_s116
                                 + 1.47466 w_k1_DIS_s116
                                 + 1.27888 w_k1_FDX_s116
                                 + 1.81106 w_k1_IBM_s116
                                 + 1.12978 w_k1_INTC_s116
                                 + 1.42902 w_k1_KO_s116 + 2.01866 w_k1_MAT_s116
                                 + 1.19621 w_k1_MSFT_s116
                                 + 1.47494 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.967853 w_k1_ADM_s116
                                 + 0.849101 w_k1_AXP_s116
                                 + 0.959271 w_k1_DIS_s116
                                 + 1.01199 w_k1_FDX_s116
                                 + 0.932642 w_k1_IBM_s116
                                 + 0.991495 w_k1_INTC_s116
                                 + 0.887411 w_k1_KO_s116
                                 + 0.832358 w_k1_MAT_s116
                                 + 0.911459 w_k1_MSFT_s116
                                 + 1.02152 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.799732 w_k1_ADM_s116
                                 + 0.756208 w_k1_AXP_s116
                                 + 0.837869 w_k1_DIS_s116
                                 + 0.867217 w_k1_FDX_s116
                                 + 0.876089 w_k1_IBM_s116
                                 + 1.07334 w_k1_INTC_s116
                                 + 0.758867 w_k1_KO_s116
                                 + 0.736829 w_k1_MAT_s116
                                 + 0.880152 w_k1_MSFT_s116
                                 + 0.94759 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07438 w_k1_ADM_s116
                                 + 1.23483 w_k1_AXP_s116
                                 + 1.08216 w_k1_DIS_s116
                                 + 1.16917 w_k1_FDX_s116
                                 + 0.727118 w_k1_IBM_s116
                                 + 1.1605 w_k1_INTC_s116
                                 + 0.807109 w_k1_KO_s116
                                 + 0.942486 w_k1_MAT_s116
                                 + 1.09572 w_k1_MSFT_s116
                                 + 1.10045 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.911745 w_k1_ADM_s116
                                 + 0.914744 w_k1_AXP_s116
                                 + 0.691701 w_k1_DIS_s116
                                 + 0.789986 w_k1_FDX_s116
                                 + 1.09387 w_k1_IBM_s116
                                 + 0.823664 w_k1_INTC_s116
                                 + 0.763881 w_k1_KO_s116
                                 + 1.03038 w_k1_MAT_s116
                                 + 0.817138 w_k1_MSFT_s116
                                 + 0.962142 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.77407 w_k1_ADM_s116
                                 + 0.639647 w_k1_AXP_s116
                                 + 0.558201 w_k1_DIS_s116
                                 + 0.761314 w_k1_FDX_s116
                                 + 0.648225 w_k1_IBM_s116
                                 + 0.867868 w_k1_INTC_s116
                                 + 0.598177 w_k1_KO_s116
                                 + 0.530926 w_k1_MAT_s116
                                 + 0.712552 w_k1_MSFT_s116
                                 + 0.849453 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.68499 w_k1_ADM_s116
                                 + 1.42921 w_k1_AXP_s116 + 1.8096 w_k1_DIS_s116
                                 + 1.52391 w_k1_FDX_s116
                                 + 1.18675 w_k1_IBM_s116
                                 + 1.21637 w_k1_INTC_s116
                                 + 1.34531 w_k1_KO_s116 + 1.31561 w_k1_MAT_s116
                                 + 1.26939 w_k1_MSFT_s116
                                 + 1.31754 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.05729 w_k1_ADM_s117
                                 + 1.43827 w_k1_AXP_s117
                                 + 1.24619 w_k1_DIS_s117
                                 + 1.25466 w_k1_FDX_s117
                                 + 1.12825 w_k1_IBM_s117
                                 + 1.01792 w_k1_INTC_s117
                                 + 1.09684 w_k1_KO_s117 + 1.94227 w_k1_MAT_s117
                                 + 1.19062 w_k1_MSFT_s117
                                 + 1.09153 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.807526 w_k1_ADM_s117
                                 + 1.03047 w_k1_AXP_s117
                                 + 0.820659 w_k1_DIS_s117
                                 + 0.934341 w_k1_FDX_s117
                                 + 1.05027 w_k1_IBM_s117
                                 + 0.991503 w_k1_INTC_s117
                                 + 0.978526 w_k1_KO_s117 + 1.5698 w_k1_MAT_s117
                                 + 0.885039 w_k1_MSFT_s117
                                 + 0.964593 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.84159 w_k1_ADM_s117
                                 + 0.734831 w_k1_AXP_s117
                                 + 0.704052 w_k1_DIS_s117
                                 + 0.878867 w_k1_FDX_s117
                                 + 0.596582 w_k1_IBM_s117
                                 + 0.873349 w_k1_INTC_s117
                                 + 0.667281 w_k1_KO_s117
                                 + 0.452332 w_k1_MAT_s117
                                 + 0.780508 w_k1_MSFT_s117
                                 + 0.876252 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.17278 w_k1_ADM_s117
                                 + 1.16805 w_k1_AXP_s117
                                 + 1.17655 w_k1_DIS_s117
                                 + 1.20281 w_k1_FDX_s117
                                 + 0.843454 w_k1_IBM_s117
                                 + 0.911515 w_k1_INTC_s117
                                 + 0.958173 w_k1_KO_s117
                                 + 1.06929 w_k1_MAT_s117
                                 + 0.999922 w_k1_MSFT_s117
                                 + 0.954535 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.752205 w_k1_ADM_s117
                                 + 0.700785 w_k1_AXP_s117
                                 + 0.755668 w_k1_DIS_s117
                                 + 0.843161 w_k1_FDX_s117
                                 + 0.757203 w_k1_IBM_s117
                                 + 0.839853 w_k1_INTC_s117
                                 + 0.714346 w_k1_KO_s117
                                 + 0.926622 w_k1_MAT_s117
                                 + 0.910287 w_k1_MSFT_s117
                                 + 0.852478 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.954717 w_k1_ADM_s117
                                 + 0.809663 w_k1_AXP_s117
                                 + 1.06536 w_k1_DIS_s117
                                 + 0.987897 w_k1_FDX_s117
                                 + 0.841579 w_k1_IBM_s117
                                 + 0.935688 w_k1_INTC_s117
                                 + 1.07614 w_k1_KO_s117
                                 + 0.903674 w_k1_MAT_s117
                                 + 1.12844 w_k1_MSFT_s117
                                 + 1.03136 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.27403 w_k1_ADM_s117
                                 + 1.39202 w_k1_AXP_s117
                                 + 1.55767 w_k1_DIS_s117 + 1.2598 w_k1_FDX_s117
                                 + 1.63273 w_k1_IBM_s117
                                 + 1.0081 w_k1_INTC_s117 + 1.35925 w_k1_KO_s117
                                 + 1.98685 w_k1_MAT_s117
                                 + 1.15543 w_k1_MSFT_s117
                                 + 1.16619 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.573213 w_k1_ADM_s117
                                 + 0.568671 w_k1_AXP_s117
                                 + 0.557961 w_k1_DIS_s117
                                 + 0.65827 w_k1_FDX_s117
                                 + 0.826465 w_k1_IBM_s117
                                 + 0.737654 w_k1_INTC_s117
                                 + 0.725852 w_k1_KO_s117
                                 + 0.807032 w_k1_MAT_s117
                                 + 0.698802 w_k1_MSFT_s117
                                 + 0.787781 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00392381615553651 w_k1_ADM_s000
                                 + 0.00683368465845716 w_k1_ADM_s001
                                 + 0.00670131168326339 w_k1_ADM_s002
                                 + 0.00900924917698805 w_k1_ADM_s003
                                 + 0.0032749128855194 w_k1_ADM_s004
                                 + 0.00382312463376926 w_k1_ADM_s005
                                 + 0.00227643418117542 w_k1_ADM_s006
                                 + 0.00956238167774969 w_k1_ADM_s007
                                 + 0.0311190671437678 w_k1_ADM_s010
                                 + 0.012128261308931 w_k1_ADM_s011
                                 + 0.0316505323153829 w_k1_ADM_s012
                                 + 0.0192118848561235 w_k1_ADM_s013
                                 + 0.0531961237725416 w_k1_ADM_s014
                                 + 0.0120561832268779 w_k1_ADM_s015
                                 + 0.0168565483460188 w_k1_ADM_s016
                                 + 0.0185819489410654 w_k1_ADM_s017
                                 + 0.0413449079934288 w_k1_ADM_s100
                                 + 0.025465099823387 w_k1_ADM_s101
                                 + 0.037547771027552 w_k1_ADM_s102
                                 + 0.0284130496350643 w_k1_ADM_s103
                                 + 0.0318166920183469 w_k1_ADM_s104
                                 + 0.0540462674936338 w_k1_ADM_s105
                                 + 0.0284393848329116 w_k1_ADM_s106
                                 + 0.0182315578172262 w_k1_ADM_s107
                                 + 0.0964927810577963 w_k1_ADM_s110
                                 + 0.0276488398894029 w_k1_ADM_s111
                                 + 0.0609635612746978 w_k1_ADM_s112
                                 + 0.0346834086852887 w_k1_ADM_s113
                                 + 0.0762940977488617 w_k1_ADM_s114
                                 + 0.098988861976748 w_k1_ADM_s115
                                 + 0.0518746638507087 w_k1_ADM_s116
                                 + 0.0844933644540589 w_k1_ADM_s117
                                 + 0.00349458946370046 w_k1_AXP_s000
                                 + 0.00681742927289614 w_k1_AXP_s001
                                 + 0.00753486936669411 w_k1_AXP_s002
                                 + 0.00843578680762977 w_k1_AXP_s003
                                 + 0.00427775886424842 w_k1_AXP_s004
                                 + 0.00386818992212377 w_k1_AXP_s005
                                 + 0.00255225860224802 w_k1_AXP_s006
                                 + 0.00898406513446566 w_k1_AXP_s007
                                 + 0.0333933554626895 w_k1_AXP_s010
                                 + 0.0134594139880674 w_k1_AXP_s011
                                 + 0.0282677755985422 w_k1_AXP_s012
                                 + 0.0217348116066318 w_k1_AXP_s013
                                 + 0.0485657096485483 w_k1_AXP_s014
                                 + 0.0137563831053884 w_k1_AXP_s015
                                 + 0.0135083769395864 w_k1_AXP_s016
                                 + 0.0207276889048235 w_k1_AXP_s017
                                 + 0.0395098938076787 w_k1_AXP_s100
                                 + 0.027553881962058 w_k1_AXP_s101
                                 + 0.032543872745532 w_k1_AXP_s102
                                 + 0.0245559025186236 w_k1_AXP_s103
                                 + 0.0359580487890515 w_k1_AXP_s104
                                 + 0.0535022555455602 w_k1_AXP_s105
                                 + 0.0307442214004324 w_k1_AXP_s106
                                 + 0.0217132356533971 w_k1_AXP_s107
                                 + 0.0786779289290836 w_k1_AXP_s110
                                 + 0.0307826389628227 w_k1_AXP_s111
                                 + 0.0595617839279175 w_k1_AXP_s112
                                 + 0.0380884964780406 w_k1_AXP_s113
                                 + 0.0779890031316309 w_k1_AXP_s114
                                 + 0.0934751899074718 w_k1_AXP_s115
                                 + 0.0517009308278935 w_k1_AXP_s116
                                 + 0.0918888715377929 w_k1_AXP_s117
                                 + 0.00371912360972292 w_k1_DIS_s000
                                 + 0.00662068614787305 w_k1_DIS_s001
                                 + 0.00734211784134103 w_k1_DIS_s002
                                 + 0.00916091607256385 w_k1_DIS_s003
                                 + 0.00349328583075057 w_k1_DIS_s004
                                 + 0.00407994302213956 w_k1_DIS_s005
                                 + 0.00231338449491374 w_k1_DIS_s006
                                 + 0.0100784239786747 w_k1_DIS_s007
                                 + 0.0316535664478771 w_k1_DIS_s010
                                 + 0.0127416812281891 w_k1_DIS_s011
                                 + 0.0290408217592987 w_k1_DIS_s012
                                 + 0.0209562276762741 w_k1_DIS_s013
                                 + 0.0529398153374483 w_k1_DIS_s014
                                 + 0.0146920543201232 w_k1_DIS_s015
                                 + 0.0151886202276443 w_k1_DIS_s016
                                 + 0.0214191667828244 w_k1_DIS_s017
                                 + 0.0370989964958777 w_k1_DIS_s100
                                 + 0.0266108954279825 w_k1_DIS_s101
                                 + 0.0332940607823839 w_k1_DIS_s102
                                 + 0.0232064686319709 w_k1_DIS_s103
                                 + 0.0366182470354359 w_k1_DIS_s104
                                 + 0.0541493116744383 w_k1_DIS_s105
                                 + 0.0307950322336232 w_k1_DIS_s106
                                 + 0.0197294752755888 w_k1_DIS_s107
                                 + 0.0872555092409134 w_k1_DIS_s110
                                 + 0.0301346203594114 w_k1_DIS_s111
                                 + 0.0577888176755774 w_k1_DIS_s112
                                 + 0.0384596563547779 w_k1_DIS_s113
                                 + 0.0824420181951379 w_k1_DIS_s114
                                 + 0.094251906925632 w_k1_DIS_s115
                                 + 0.0537513069193241 w_k1_DIS_s116
                                 + 0.0912230925840826 w_k1_DIS_s117
                                 + 0.00395660338240298 w_k1_FDX_s000
                                 + 0.00697929430721146 w_k1_FDX_s001
                                 + 0.00740024448205196 w_k1_FDX_s002
                                 + 0.00798633262598373 w_k1_FDX_s003
                                 + 0.00354966973494592 w_k1_FDX_s004
                                 + 0.00400113998302201 w_k1_FDX_s005
                                 + 0.002317474133433 w_k1_FDX_s006
                                 + 0.00926239179999751 w_k1_FDX_s007
                                 + 0.0325856391692363 w_k1_FDX_s010
                                 + 0.0123051505222774 w_k1_FDX_s011
                                 + 0.0297473099043884 w_k1_FDX_s012
                                 + 0.0208972877614592 w_k1_FDX_s013
                                 + 0.0485184445173126 w_k1_FDX_s014
                                 + 0.0136117444180853 w_k1_FDX_s015
                                 + 0.0142464585540226 w_k1_FDX_s016
                                 + 0.0193598897491863 w_k1_FDX_s017
                                 + 0.0370068871515209 w_k1_FDX_s100
                                 + 0.0266776595694002 w_k1_FDX_s101
                                 + 0.0303059731401088 w_k1_FDX_s102
                                 + 0.0216984373718736 w_k1_FDX_s103
                                 + 0.0348831754216355 w_k1_FDX_s104
                                 + 0.0544671028060529 w_k1_FDX_s105
                                 + 0.0294101285381793 w_k1_FDX_s106
                                 + 0.0192496549961257 w_k1_FDX_s107
                                 + 0.0855245028589163 w_k1_FDX_s110
                                 + 0.0293790589085406 w_k1_FDX_s111
                                 + 0.0572867930975917 w_k1_FDX_s112
                                 + 0.0398601590767979 w_k1_FDX_s113
                                 + 0.0774601553865627 w_k1_FDX_s114
                                 + 0.0966468583127287 w_k1_FDX_s115
                                 + 0.0525683535899237 w_k1_FDX_s116
                                 + 0.0913095897782995 w_k1_FDX_s117
                                 + 0.00400778404293093 w_k1_IBM_s000
                                 + 0.00666227674546025 w_k1_IBM_s001
                                 + 0.00805321553473992 w_k1_IBM_s002
                                 + 0.00843745388488511 w_k1_IBM_s003
                                 + 0.00342130667812406 w_k1_IBM_s004
                                 + 0.00390613270228005 w_k1_IBM_s005
                                 + 0.00257019078757785 w_k1_IBM_s006
                                 + 0.00950223961641065 w_k1_IBM_s007
                                 + 0.0318151500266663 w_k1_IBM_s010
                                 + 0.0125184101101736 w_k1_IBM_s011
                                 + 0.0304644671430534 w_k1_IBM_s012
                                 + 0.0209598258945656 w_k1_IBM_s013
                                 + 0.0506534489474153 w_k1_IBM_s014
                                 + 0.0138478594103533 w_k1_IBM_s015
                                 + 0.0133198637421115 w_k1_IBM_s016
                                 + 0.0201377285693504 w_k1_IBM_s017
                                 + 0.0386843530888349 w_k1_IBM_s100
                                 + 0.0281406034116386 w_k1_IBM_s101
                                 + 0.0307173141740411 w_k1_IBM_s102
                                 + 0.0232601529717535 w_k1_IBM_s103
                                 + 0.0342071251236198 w_k1_IBM_s104
                                 + 0.0529828167451429 w_k1_IBM_s105
                                 + 0.02906505600172 w_k1_IBM_s106
                                 + 0.019248837749474 w_k1_IBM_s107
                                 + 0.0855314572368426 w_k1_IBM_s110
                                 + 0.0288204038611794 w_k1_IBM_s111
                                 + 0.0578292134672261 w_k1_IBM_s112
                                 + 0.0397945842826655 w_k1_IBM_s113
                                 + 0.0774665107318568 w_k1_IBM_s114
                                 + 0.0956868217846887 w_k1_IBM_s115
                                 + 0.0536756364620823 w_k1_IBM_s116
                                 + 0.0884533683061207 w_k1_IBM_s117
                                 + 0.00460534672658166 w_k1_INTC_s000
                                 + 0.00724668613196152 w_k1_INTC_s001
                                 + 0.00720882234627682 w_k1_INTC_s002
                                 + 0.00796936090062975 w_k1_INTC_s003
                                 + 0.00356065452708758 w_k1_INTC_s004
                                 + 0.00481347178887847 w_k1_INTC_s005
                                 + 0.00241010300042925 w_k1_INTC_s006
                                 + 0.00910641033716189 w_k1_INTC_s007
                                 + 0.0318539731159094 w_k1_INTC_s010
                                 + 0.0111202225440547 w_k1_INTC_s011
                                 + 0.0266198345190602 w_k1_INTC_s012
                                 + 0.0173354527441728 w_k1_INTC_s013
                                 + 0.0567207318740909 w_k1_INTC_s014
                                 + 0.0139768493976959 w_k1_INTC_s015
                                 + 0.0138853738006716 w_k1_INTC_s016
                                 + 0.0170565583459903 w_k1_INTC_s017
                                 + 0.03869386385702 w_k1_INTC_s100
                                 + 0.0265901396311308 w_k1_INTC_s101
                                 + 0.0278897844261028 w_k1_INTC_s102
                                 + 0.0219917193855552 w_k1_INTC_s103
                                 + 0.0313037331499321 w_k1_INTC_s104
                                 + 0.0559510047386916 w_k1_INTC_s105
                                 + 0.0304826368552029 w_k1_INTC_s106
                                 + 0.0196839065010925 w_k1_INTC_s107
                                 + 0.0736933432020394 w_k1_INTC_s110
                                 + 0.0265424777655112 w_k1_INTC_s111
                                 + 0.0564934526854286 w_k1_INTC_s112
                                 + 0.039890873208924 w_k1_INTC_s113
                                 + 0.0753103545549055 w_k1_INTC_s114
                                 + 0.0920932126865467 w_k1_INTC_s115
                                 + 0.0540771968275031 w_k1_INTC_s116
                                 + 0.0820149254294064 w_k1_INTC_s117
                                 + 0.00377928226142964 w_k1_KO_s000
                                 + 0.00649357136024965 w_k1_KO_s001
                                 + 0.00756167434047735 w_k1_KO_s002
                                 + 0.00993770721968213 w_k1_KO_s003
                                 + 0.00378724101085482 w_k1_KO_s004
                                 + 0.0039532579570731 w_k1_KO_s005
                                 + 0.0023184943430299 w_k1_KO_s006
                                 + 0.00976742885660378 w_k1_KO_s007
                                 + 0.029929429728913 w_k1_KO_s010
                                 + 0.0125923191035653 w_k1_KO_s011
                                 + 0.0275435503208601 w_k1_KO_s012
                                 + 0.0205316831710038 w_k1_KO_s013
                                 + 0.0596747114232927 w_k1_KO_s014
                                 + 0.0128669509632332 w_k1_KO_s015
                                 + 0.0145539410236804 w_k1_KO_s016
                                 + 0.0186056838287694 w_k1_KO_s017
                                 + 0.0393745605988313 w_k1_KO_s100
                                 + 0.02615427217988 w_k1_KO_s101
                                 + 0.033713438807076 w_k1_KO_s102
                                 + 0.0241518724437137 w_k1_KO_s103
                                 + 0.0373945557209919 w_k1_KO_s104
                                 + 0.0562670623001734 w_k1_KO_s105
                                 + 0.0313521016353651 w_k1_KO_s106
                                 + 0.0173714585980968 w_k1_KO_s107
                                 + 0.0803415004418595 w_k1_KO_s110
                                 + 0.0281425771335479 w_k1_KO_s111
                                 + 0.05868730448431 w_k1_KO_s112
                                 + 0.0358048365563538 w_k1_KO_s113
                                 + 0.0702720405872781 w_k1_KO_s114
                                 + 0.0986540424335596 w_k1_KO_s115
                                 + 0.0488489197285187 w_k1_KO_s116
                                 + 0.086802552089434 w_k1_KO_s117
                                 + 0.00406050627102985 w_k1_MAT_s000
                                 + 0.00837610001511867 w_k1_MAT_s001
                                 + 0.00852396887045153 w_k1_MAT_s002
                                 + 0.00790234169932318 w_k1_MAT_s003
                                 + 0.0043954911540242 w_k1_MAT_s004
                                 + 0.00343915111367564 w_k1_MAT_s005
                                 + 0.00269512733099407 w_k1_MAT_s006
                                 + 0.00856931621601087 w_k1_MAT_s007
                                 + 0.0340033543531776 w_k1_MAT_s010
                                 + 0.0154283131364135 w_k1_MAT_s011
                                 + 0.0290983955419571 w_k1_MAT_s012
                                 + 0.0248462331267087 w_k1_MAT_s013
                                 + 0.0647860897103462 w_k1_MAT_s014
                                 + 0.0135873514286292 w_k1_MAT_s015
                                 + 0.010887311846313 w_k1_MAT_s016
                                 + 0.0223196871196067 w_k1_MAT_s017
                                 + 0.0401884891719955 w_k1_MAT_s100
                                 + 0.0277790711458613 w_k1_MAT_s101
                                 + 0.0308018027422475 w_k1_MAT_s102
                                 + 0.0216614519931322 w_k1_MAT_s103
                                 + 0.0377048955960898 w_k1_MAT_s104
                                 + 0.0565974226581559 w_k1_MAT_s105
                                 + 0.0319405174395952 w_k1_MAT_s106
                                 + 0.0201530630708352 w_k1_MAT_s107
                                 + 0.0861103772874215 w_k1_MAT_s110
                                 + 0.0269157923113571 w_k1_MAT_s111
                                 + 0.0597223445214978 w_k1_MAT_s112
                                 + 0.0416483046661822 w_k1_MAT_s113
                                 + 0.0925625648140971 w_k1_MAT_s114
                                 + 0.108450861531519 w_k1_MAT_s115
                                 + 0.0519282088561831 w_k1_MAT_s116
                                 + 0.11697286609584 w_k1_MAT_s117
                                 + 0.0044760102700724 w_k1_MSFT_s000
                                 + 0.00688506574014243 w_k1_MSFT_s001
                                 + 0.00745187108286348 w_k1_MSFT_s002
                                 + 0.0084908249057894 w_k1_MSFT_s003
                                 + 0.00323073900249191 w_k1_MSFT_s004
                                 + 0.00421004687021026 w_k1_MSFT_s005
                                 + 0.0022553457250004 w_k1_MSFT_s006
                                 + 0.0105001981337043 w_k1_MSFT_s007
                                 + 0.030816307526394 w_k1_MSFT_s010
                                 + 0.0123185339350261 w_k1_MSFT_s011
                                 + 0.0281293060534298 w_k1_MSFT_s012
                                 + 0.0192903988032823 w_k1_MSFT_s013
                                 + 0.0505581451147475 w_k1_MSFT_s014
                                 + 0.0131483968489504 w_k1_MSFT_s015
                                 + 0.0144418136584757 w_k1_MSFT_s016
                                 + 0.0184564089776737 w_k1_MSFT_s017
                                 + 0.0354764349310991 w_k1_MSFT_s100
                                 + 0.0275299144308767 w_k1_MSFT_s101
                                 + 0.0325586577902692 w_k1_MSFT_s102
                                 + 0.0221660108717109 w_k1_MSFT_s103
                                 + 0.0338383155437656 w_k1_MSFT_s104
                                 + 0.0504597162241087 w_k1_MSFT_s105
                                 + 0.030117814870682 w_k1_MSFT_s106
                                 + 0.0169895894215946 w_k1_MSFT_s107
                                 + 0.0809954174821448 w_k1_MSFT_s110
                                 + 0.0282697617204865 w_k1_MSFT_s111
                                 + 0.0645504067630332 w_k1_MSFT_s112
                                 + 0.0394023124803899 w_k1_MSFT_s113
                                 + 0.0754208394385711 w_k1_MSFT_s114
                                 + 0.0939455851661662 w_k1_MSFT_s115
                                 + 0.0506774586943095 w_k1_MSFT_s116
                                 + 0.0875866930770785 w_k1_MSFT_s117
                                 + 0.00370041607872384 w_k1_PG_s000
                                 + 0.00673936427033485 w_k1_PG_s001
                                 + 0.00755985933493234 w_k1_PG_s002
                                 + 0.00833953611203647 w_k1_PG_s003
                                 + 0.00353529167945264 w_k1_PG_s004
                                 + 0.00366495924875468 w_k1_PG_s005
                                 + 0.00237335279110308 w_k1_PG_s006
                                 + 0.00937347572215756 w_k1_PG_s007
                                 + 0.0299015998805446 w_k1_PG_s010
                                 + 0.0117897672478624 w_k1_PG_s011
                                 + 0.0324557620102107 w_k1_PG_s012
                                 + 0.0202554088358427 w_k1_PG_s013
                                 + 0.05133063626188 w_k1_PG_s014
                                 + 0.0127303336081696 w_k1_PG_s015
                                 + 0.0134786697294434 w_k1_PG_s016
                                 + 0.019590382774257 w_k1_PG_s017
                                 + 0.0369991411014228 w_k1_PG_s100
                                 + 0.0271280687412608 w_k1_PG_s101
                                 + 0.0316639274507227 w_k1_PG_s102
                                 + 0.0227924588769512 w_k1_PG_s103
                                 + 0.0355968493409445 w_k1_PG_s104
                                 + 0.0523571678872917 w_k1_PG_s105
                                 + 0.0302282212010501 w_k1_PG_s106
                                 + 0.0197472537205475 w_k1_PG_s107
                                 + 0.0838351593133713 w_k1_PG_s110
                                 + 0.0264383122230261 w_k1_PG_s111
                                 + 0.0544352505031542 w_k1_PG_s112
                                 + 0.035737395611973 w_k1_PG_s113
                                 + 0.0818130951975301 w_k1_PG_s114
                                 + 0.0946815949008037 w_k1_PG_s115
                                 + 0.056120133590339 w_k1_PG_s116
                                 + 0.0864354988817886 w_k1_PG_s117 >= 1.06
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
