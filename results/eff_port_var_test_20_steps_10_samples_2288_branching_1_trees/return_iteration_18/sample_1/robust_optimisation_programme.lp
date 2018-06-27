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
 w_k1_ADM_s0_return_constraint:  1.10436 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.9463 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.17413 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.07764 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.987424 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.01931 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.821499 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.669717 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.647814 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.05647 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.03384 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.32584 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.901925 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.966615 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.709702 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.06265 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.813312 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.891932 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.896288 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.997861 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.01331 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.48379 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.951107 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.936069 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.778208 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.60531 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.04718 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.765352 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.00385 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.29313 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.01798 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.34114 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.80079 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.72357 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.56895 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.10015 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.28538 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.02575 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24497 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.912845 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.11233 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.928927 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.872546 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.19144 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.800122 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.595053 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.587377 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.03434 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.2802 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.69471 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.3803 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.928401 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.684356 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.49864 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.676982 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.920584 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.836181 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.14845 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.904477 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.37858 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.974815 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.0657 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.616248 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.79738 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.31665 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.695347 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.34916 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.67024 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.930638 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.18328 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.610566 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.918002 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.49261 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.836046 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.15859 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.883175 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.35137 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.886876 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.10814 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.816853 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.77252 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.27491 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.86106 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.521987 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.732726 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.08888 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.12616 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 2.01798 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.36596 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.00959 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.582816 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.35289 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.684024 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.05773 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.940564 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.843378 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.25605 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.92848 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.07792 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.862277 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.687493 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.95118 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.15454 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.73002 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.53016 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.6933 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.978203 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.40722 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.564474 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.791674 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.6219 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.908901 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.03026 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.774855 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20321 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.921848 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.04051 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.9894 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.915937 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.07744 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.850026 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.658635 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.776494 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0378 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.19994 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.6587 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.0711 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.0854 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.683745 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.28333 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.7782 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.924472 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.934685 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.99474 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.11039 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.59851 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.991514 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.837355 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.69188 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.61181 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.15952 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.818873 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.18358 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.47416 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.93822 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.40769 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.697859 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.786677 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.37704 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.07249 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.15388 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.883562 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15574 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.947399 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.18995 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.63721 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.805838 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.25246 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.23766 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.773564 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.785875 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.938258 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.12322 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.21012 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.78941 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.690132 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.99523 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.19586 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.872104 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.41287 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.738405 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.938103 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.00844 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.19359 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.853335 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.42291 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.907274 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.0047 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.911677 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.587901 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.36318 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.43848 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.31499 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.791703 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.800282 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.81834 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.57341 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.605948 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.988552 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.915218 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16329 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.987091 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.976928 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.996258 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.854112 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.08208 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.909551 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.840805 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.94242 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.1033 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.24375 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.49825 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.16328 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.05613 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.738919 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.12658 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.740163 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.917887 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.789123 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.810616 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.88083 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.23 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0422 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.93 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.824294 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.32296 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.0953 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.768429 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.28504 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.44019 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.937446 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.3048 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.652269 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.858748 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.12534 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.854095 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.02077 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.866228 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15505 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.960886 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.15817 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.782191 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.751054 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.25384 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.03562 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.820683 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.822513 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.972525 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.20231 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.41409 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.31595 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.830377 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.786318 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.17381 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.784452 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.24021 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.939809 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.983752 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.1618 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.34484 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.08822 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15013 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.860807 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.27053 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.10152 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.674803 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.50724 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.56641 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.04075 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.12705 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.666803 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.966815 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.36231 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.805592 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.942422 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.834353 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25755 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.927102 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.17089 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.71368 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.759653 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.26871 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.977322 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.720612 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.638447 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.801301 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.37415 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.4952 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 2.01594 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.585109 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.908559 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.36358 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.758417 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.09581 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.803082 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.14299 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.878398 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.13365 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.891045 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.64356 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.828518 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.15468 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.36014 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.601033 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.47364 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.82738 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.14075 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.785961 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.899393 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.69669 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.72874 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.644421 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.972008 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.800226 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22478 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.920728 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.988795 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.878926 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.862123 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16236 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.8988 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.728876 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.898075 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.14136 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.17034 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.57246 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16908 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.03265 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.718751 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.16131 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.804885 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.06423 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.980768 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.861492 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.03827 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.66259 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.05701 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.859009 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.873238 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.38301 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.25498 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.848178 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.26822 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.40131 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.919026 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.23005 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.714906 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.773883 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.29173 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.944525 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.05043 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.801585 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09607 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.95606 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08634 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.916032 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.887096 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.07695 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.890137 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.779943 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.829653 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.01291 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.11256 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26248 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.10061 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.932647 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.806599 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.19275 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.860623 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.07528 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03272 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.995282 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.13885 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.35754 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.987297 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.0698 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.837157 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.29136 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.10592 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.794334 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.20151 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.25803 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.941211 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.09026 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.81677 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.957665 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.30672 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.964859 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04262 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.913737 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.720000993761629 z_k1_s0000
                                 - 0.21300035427123 z_k1_s0001
                                 - 0.410001972505052 z_k1_s0002
                                 - 0.554002171257378 z_k1_s0003
                                 - 0.557998961662817 z_k1_s0004
                                 - 0.563999572778312 z_k1_s0005
                                 - 0.412002176210217 z_k1_s0006
                                 - 0.186000136154924 z_k1_s0007
                                 - 0.580000563381559 z_k1_s0010
                                 - 0.323999773082294 z_k1_s0011
                                 - 0.540000178803636 z_k1_s0012
                                 - 0.459000151983091 z_k1_s0013
                                 - 0.409000506760407 z_k1_s0014
                                 - 0.405000134102727 z_k1_s0015
                                 - 0.212000033063662 z_k1_s0016
                                 - 0.413000879418087 z_k1_s0017
                                 - 0.633000354863847 z_k1_s0020
                                 - 0.263999770369741 z_k1_s0021
                                 - 0.248000025741282 z_k1_s0022
                                 - 0.367001114415568 z_k1_s0023
                                 - 0.349000903708531 z_k1_s0024
                                 - 0.22600012787714 z_k1_s0025
                                 - 0.248000025741282 z_k1_s0026
                                 - 0.155000016088301 z_k1_s0027
                                 - 0.51900185469808 z_k1_s0030
                                 - 0.226000693960642 z_k1_s0031
                                 - 0.710002028371434 z_k1_s0032
                                 - 0.480002627353803 z_k1_s0033
                                 - 0.175000543429721 z_k1_s0034
                                 - 0.420001441431778 z_k1_s0035
                                 - 0.585002130208948 z_k1_s0036
                                 - 0.315000909573273 z_k1_s0037
                                 - 0.227000856005231 z_k1_s0040
                                 - 0.213000959367637 z_k1_s0041
                                 - 0.819001763729965 z_k1_s0042
                                 - 0.242001088575432 z_k1_s0043
                                 - 0.240001017512581 z_k1_s0044
                                 - 0.432000930319102 z_k1_s0045
                                 - 0.22500078494238 z_k1_s0046
                                 - 0.606002306368233 z_k1_s0047
                                 - 0.28799946433231 z_k1_s0050
                                 - 0.194000129979704 z_k1_s0051
                                 - 0.370999227150251 z_k1_s0052
                                 - 0.196000111204224 z_k1_s0053
                                 - 0.218001205474748 z_k1_s0054
                                 - 0.476999316050512 z_k1_s0055
                                 - 0.158000034338065 z_k1_s0056
                                 - 0.266000754863245 z_k1_s0057
                                 - 0.18400005997561 z_k1_s0060
                                 - 0.395001053230801 z_k1_s0061
                                 - 0.100000141852027 z_k1_s0062
                                 - 0.380000964297956 z_k1_s0063
                                 - 0.208999950463531 z_k1_s0064
                                 - 0.427999315881907 z_k1_s0065
                                 - 0.14500002205033 z_k1_s0066
                                 - 0.0920000299878051 z_k1_s0067
                                 - 0.659999762013933 z_k1_s0070
                                 - 0.41900054996623 z_k1_s0071
                                 - 1.04600062433003 z_k1_s0072
                                 - 0.926000667600228 z_k1_s0073
                                 - 1.13300509971766 z_k1_s0074
                                 - 0.904001677034989 z_k1_s0075
                                 - 0.417000750987779 z_k1_s0076
                                 - 0.504001020067456 z_k1_s0077
                                 - 0.236999489326192 z_k1_s0100
                                 - 0.127000353900108 z_k1_s0101
                                 - 0.218999961528972 z_k1_s0102
                                 - 0.125999904578363 z_k1_s0103
                                 - 0.172000243392432 z_k1_s0104
                                 - 0.0669999305773959 z_k1_s0105
                                 - 0.0519999317466454 z_k1_s0106
                                 - 0.0699999945435025 z_k1_s0107
                                 - 0.054999764294622 z_k1_s0110
                                 - 0.179999098273776 z_k1_s0111
                                 - 0.0819996802355046 z_k1_s0112
                                 - 0.165999020497457 z_k1_s0113
                                 - 0.223999196428445 z_k1_s0114
                                 - 0.127999540816254 z_k1_s0115
                                 - 0.0869996494986855 z_k1_s0116
                                 - 0.101999659687861 z_k1_s0117
                                 - 0.135999637253486 z_k1_s0120
                                 - 0.136999493880262 z_k1_s0121
                                 - 0.133999923999935 z_k1_s0122
                                 - 0.177999595581812 z_k1_s0123
                                 - 0.151999735283396 z_k1_s0124
                                 - 0.144999542895217 z_k1_s0125
                                 - 0.168999689940082 z_k1_s0126
                                 - 0.144999542895217 z_k1_s0127
                                 - 0.154999705247438 z_k1_s0130
                                 - 0.283998836069191 z_k1_s0131
                                 - 0.0629996587241879 z_k1_s0132
                                 - 0.0759996558383871 z_k1_s0133
                                 - 0.0989990630970256 z_k1_s0134
                                 - 0.0609997186755713 z_k1_s0135
                                 - 0.115999810604389 z_k1_s0136
                                 - 0.112999223634629 z_k1_s0137
                                 - 0.112999627391901 z_k1_s0140
                                 - 0.0839999078522611 z_k1_s0141
                                 - 0.102999842171255 z_k1_s0142
                                 - 0.12999977586665 z_k1_s0143
                                 - 0.125999690578508 z_k1_s0144
                                 - 0.174999665358974 z_k1_s0145
                                 - 0.0519999959466019 z_k1_s0146
                                 - 0.0729999729096672 z_k1_s0147
                                 - 0.0709999394494878 z_k1_s0150
                                 - 0.0509999457397851 z_k1_s0151
                                 - 0.0629999292256211 z_k1_s0152
                                 - 0.167999641401849 z_k1_s0153
                                 - 0.0969997019523613 z_k1_s0154
                                 - 0.0689997617187198 z_k1_s0155
                                 - 0.0740002060456396 z_k1_s0156
                                 - 0.0439999606813747 z_k1_s0157
                                 - 0.174999462098959 z_k1_s0160
                                 - 0.161998890231016 z_k1_s0161
                                 - 0.171998933207827 z_k1_s0162
                                 - 0.092999625687203 z_k1_s0163
                                 - 0.278998877061609 z_k1_s0164
                                 - 0.0899996127941598 z_k1_s0165
                                 - 0.138999565381486 z_k1_s0166
                                 - 0.179999483587364 z_k1_s0167
                                 - 0.077999863215668 z_k1_s0170
                                 - 0.0889998580281273 z_k1_s0171
                                 - 0.0930000061416782 z_k1_s0172
                                 - 0.0920001066132256 z_k1_s0173
                                 - 0.0889998580281273 z_k1_s0174
                                 - 0.0410000056646094 z_k1_s0175
                                 - 0.0319999749089774 z_k1_s0176
                                 - 0.035999958022606 z_k1_s0177
                                 - 1.23000248953765 z_k1_s1000
                                 - 1.35899796988069 z_k1_s1001
                                 - 0.434000004698072 z_k1_s1002
                                 - 0.567999847296598 z_k1_s1003
                                 - 0.624999830782524 z_k1_s1004
                                 - 0.558999565314686 z_k1_s1005
                                 - 1.47100132878897 z_k1_s1006
                                 - 0.512999851373339 z_k1_s1007
                                 - 0.458996564804054 z_k1_s1010
                                 - 0.674998649994911 z_k1_s1011
                                 - 0.660996766793315 z_k1_s1012
                                 - 0.384998747202242 z_k1_s1013
                                 - 0.928996856759796 z_k1_s1014
                                 - 0.592996609787295 z_k1_s1015
                                 - 0.572998414485881 z_k1_s1016
                                 - 0.219999476757849 z_k1_s1017
                                 - 0.499998866817342 z_k1_s1020
                                 - 0.246999020889539 z_k1_s1021
                                 - 0.347998677627126 z_k1_s1022
                                 - 0.203999080223409 z_k1_s1023
                                 - 0.849996167597537 z_k1_s1024
                                 - 0.879996480804967 z_k1_s1025
                                 - 0.560998233011152 z_k1_s1026
                                 - 0.221998886949478 z_k1_s1027
                                 - 0.184999729119684 z_k1_s1030
                                 - 0.376999557802709 z_k1_s1031
                                 - 0.356999273336023 z_k1_s1032
                                 - 0.0849999651846323 z_k1_s1033
                                 - 0.363999787498958 z_k1_s1034
                                 - 0.236999639533879 z_k1_s1035
                                 - 0.13299981870549 z_k1_s1036
                                 - 0.335000618821048 z_k1_s1037
                                 - 0.8050004253862 z_k1_s1040
                                 - 0.350999792153858 z_k1_s1041
                                 - 1.04800054178478 z_k1_s1042
                                 - 0.672998512109708 z_k1_s1043
                                 - 0.496001769347092 z_k1_s1044
                                 - 0.310999930148182 z_k1_s1045
                                 - 0.668000160832457 z_k1_s1046
                                 - 0.481999618177183 z_k1_s1047
                                 - 0.870999916658467 z_k1_s1050
                                 - 0.448997819524332 z_k1_s1051
                                 - 0.4310006712642 z_k1_s1052
                                 - 0.321999716420055 z_k1_s1053
                                 - 0.348999502205796 z_k1_s1054
                                 - 0.292999631828307 z_k1_s1055
                                 - 0.404998633784096 z_k1_s1056
                                 - 0.573998948418957 z_k1_s1057
                                 - 0.203999824006409 z_k1_s1060
                                 - 0.158999974565239 z_k1_s1061
                                 - 0.4540009756009 z_k1_s1062
                                 - 0.567999385025745 z_k1_s1063
                                 - 0.311999903845018 z_k1_s1064
                                 - 0.408000628412375 z_k1_s1065
                                 - 0.226999997600672 z_k1_s1066
                                 - 0.118999917762063 z_k1_s1067
                                 - 0.0789999457482656 z_k1_s1070
                                 - 0.120999836609438 z_k1_s1071
                                 - 0.324999680806239 z_k1_s1072
                                 - 0.100999941113596 z_k1_s1073
                                 - 0.215999478114568 z_k1_s1074
                                 - 0.172999537318651 z_k1_s1075
                                 - 0.280000191740356 z_k1_s1076
                                 - 0.0839999196222261 z_k1_s1077
                                 - 0.495000754193934 z_k1_s1100
                                 - 1.12800156633257 z_k1_s1101
                                 - 0.538001054529381 z_k1_s1102
                                 - 0.80999952058702 z_k1_s1103
                                 - 0.396000603355147 z_k1_s1104
                                 - 1.66000297346934 z_k1_s1105
                                 - 0.477000555413849 z_k1_s1106
                                 - 0.778999143337923 z_k1_s1107
                                 - 0.834001319127022 z_k1_s1110
                                 - 0.212999848490724 z_k1_s1111
                                 - 0.390001392109743 z_k1_s1112
                                 - 0.300999895510364 z_k1_s1113
                                 - 0.203000079893033 z_k1_s1114
                                 - 0.812000319572132 z_k1_s1115
                                 - 0.516001565560588 z_k1_s1116
                                 - 0.322999817465276 z_k1_s1117
                                 - 0.320000333918602 z_k1_s1120
                                 - 1.10300157070073 z_k1_s1121
                                 - 0.649000224903612 z_k1_s1122
                                 - 1.0109978397987 z_k1_s1123
                                 - 1.25000150811956 z_k1_s1124
                                 - 0.358000202421416 z_k1_s1125
                                 - 1.23299734610465 z_k1_s1126
                                 - 0.595999790623298 z_k1_s1127
                                 - 0.237000305443782 z_k1_s1130
                                 - 0.0719999573184415 z_k1_s1131
                                 - 0.147999793349087 z_k1_s1132
                                 - 0.260999577717003 z_k1_s1133
                                 - 0.114999866423937 z_k1_s1134
                                 - 0.146000567492577 z_k1_s1135
                                 - 0.163000408769238 z_k1_s1136
                                 - 0.285000276989409 z_k1_s1137
                                 - 1.10299971565459 z_k1_s1140
                                 - 0.242999745729882 z_k1_s1141
                                 - 0.250999538601195 z_k1_s1142
                                 - 0.377999179402831 z_k1_s1143
                                 - 0.971997070123096 z_k1_s1144
                                 - 0.28399958081869 z_k1_s1145
                                 - 1.19699931423873 z_k1_s1146
                                 - 0.353999322589783 z_k1_s1147
                                 - 0.745000318754055 z_k1_s1150
                                 - 0.43800057448267 z_k1_s1151
                                 - 1.08100139150857 z_k1_s1152
                                 - 0.433000377254579 z_k1_s1153
                                 - 0.627001754657711 z_k1_s1154
                                 - 0.607000042944344 z_k1_s1155
                                 - 0.390000618975097 z_k1_s1156
                                 - 0.293000391674032 z_k1_s1157
                                 - 0.417000991784572 z_k1_s1160
                                 - 0.62399996019132 z_k1_s1161
                                 - 0.774002666321894 z_k1_s1162
                                 - 0.646003240258604 z_k1_s1163
                                 - 0.314000839939725 z_k1_s1164
                                 - 0.201000360585757 z_k1_s1165
                                 - 0.838000172847315 z_k1_s1166
                                 - 0.599000244671632 z_k1_s1167
                                 - 0.44000097817626 z_k1_s1170
                                 - 0.419000949521303 z_k1_s1171
                                 - 1.15000081362664 z_k1_s1172
                                 - 1.14300344858181 z_k1_s1173
                                 - 0.608003851922734 z_k1_s1174
                                 - 0.345000772989356 z_k1_s1175
                                 - 0.757000655156279 z_k1_s1176
                                 - 0.427001111552153 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.968015 w_k1_ADM_s000
                                 + 0.944477 w_k1_AXP_s000
                                 + 0.915392 w_k1_DIS_s000
                                 + 0.881749 w_k1_FDX_s000
                                 + 0.894388 w_k1_IBM_s000
                                 + 0.957166 w_k1_INTC_s000
                                 + 0.837754 w_k1_KO_s000
                                 + 0.794102 w_k1_MAT_s000
                                 + 0.989705 w_k1_MSFT_s000
                                 + 0.913506 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.02315 w_k1_ADM_s000
                                 + 1.28156 w_k1_AXP_s000
                                 + 1.58646 w_k1_DIS_s000
                                 + 1.14955 w_k1_FDX_s000
                                 + 0.992139 w_k1_IBM_s000
                                 + 1.42394 w_k1_INTC_s000
                                 + 1.24153 w_k1_KO_s000 + 1.21781 w_k1_MAT_s000
                                 + 1.33912 w_k1_MSFT_s000
                                 + 1.32205 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.0963 w_k1_ADM_s000
                                 + 1.47872 w_k1_AXP_s000
                                 + 1.63316 w_k1_DIS_s000
                                 + 1.20527 w_k1_FDX_s000
                                 + 1.47834 w_k1_IBM_s000
                                 + 1.3455 w_k1_INTC_s000 + 1.55219 w_k1_KO_s000
                                 + 1.55336 w_k1_MAT_s000
                                 + 1.33582 w_k1_MSFT_s000
                                 + 1.40246 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.2197 w_k1_ADM_s000
                                 + 1.19856 w_k1_AXP_s000
                                 + 1.18665 w_k1_DIS_s000
                                 + 1.22566 w_k1_FDX_s000
                                 + 0.857053 w_k1_IBM_s000
                                 + 1.1855 w_k1_INTC_s000
                                 + 0.921056 w_k1_KO_s000
                                 + 0.991445 w_k1_MAT_s000
                                 + 1.23264 w_k1_MSFT_s000
                                 + 1.04282 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.86519 w_k1_ADM_s000
                                 + 0.860318 w_k1_AXP_s000
                                 + 0.691984 w_k1_DIS_s000
                                 + 0.895761 w_k1_FDX_s000
                                 + 0.672525 w_k1_IBM_s000
                                 + 1.02396 w_k1_INTC_s000
                                 + 0.72236 w_k1_KO_s000
                                 + 0.768954 w_k1_MAT_s000
                                 + 0.930836 w_k1_MSFT_s000
                                 + 0.852916 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.910696 w_k1_ADM_s000
                                 + 1.1537 w_k1_AXP_s000
                                 + 0.897832 w_k1_DIS_s000
                                 + 0.830472 w_k1_FDX_s000
                                 + 1.31107 w_k1_IBM_s000
                                 + 0.905535 w_k1_INTC_s000
                                 + 1.03156 w_k1_KO_s000 + 1.58295 w_k1_MAT_s000
                                 + 0.922831 w_k1_MSFT_s000
                                 + 1.00059 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.940446 w_k1_ADM_s000
                                 + 1.07982 w_k1_AXP_s000
                                 + 0.864536 w_k1_DIS_s000
                                 + 0.967341 w_k1_FDX_s000
                                 + 1.03845 w_k1_IBM_s000
                                 + 1.13636 w_k1_INTC_s000
                                 + 0.935069 w_k1_KO_s000
                                 + 1.00535 w_k1_MAT_s000
                                 + 0.912144 w_k1_MSFT_s000
                                 + 1.11275 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.23294 w_k1_ADM_s000
                                 + 1.58376 w_k1_AXP_s000
                                 + 1.69942 w_k1_DIS_s000
                                 + 1.38572 w_k1_FDX_s000
                                 + 1.11715 w_k1_IBM_s000
                                 + 1.28485 w_k1_INTC_s000
                                 + 1.22313 w_k1_KO_s000 + 1.31849 w_k1_MAT_s000
                                 + 1.291 w_k1_MSFT_s000 + 1.22996 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.889683 w_k1_ADM_s001
                                 + 0.74075 w_k1_AXP_s001
                                 + 0.867954 w_k1_DIS_s001
                                 + 0.889216 w_k1_FDX_s001
                                 + 0.729605 w_k1_IBM_s001
                                 + 0.828901 w_k1_INTC_s001
                                 + 0.682367 w_k1_KO_s001
                                 + 0.475089 w_k1_MAT_s001
                                 + 0.846421 w_k1_MSFT_s001
                                 + 0.852788 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.05152 w_k1_ADM_s001
                                 + 1.24432 w_k1_AXP_s001
                                 + 1.01052 w_k1_DIS_s001
                                 + 0.987406 w_k1_FDX_s001
                                 + 1.35515 w_k1_IBM_s001
                                 + 0.691908 w_k1_INTC_s001
                                 + 0.879948 w_k1_KO_s001
                                 + 1.46051 w_k1_MAT_s001
                                 + 0.818125 w_k1_MSFT_s001
                                 + 0.983385 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.954446 w_k1_ADM_s001
                                 + 1.32281 w_k1_AXP_s001
                                 + 1.35986 w_k1_DIS_s001
                                 + 1.03526 w_k1_FDX_s001
                                 + 1.69552 w_k1_IBM_s001
                                 + 1.07557 w_k1_INTC_s001
                                 + 1.21726 w_k1_KO_s001 + 1.76646 w_k1_MAT_s001
                                 + 1.06159 w_k1_MSFT_s001
                                 + 1.24431 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.906074 w_k1_ADM_s001
                                 + 1.05219 w_k1_AXP_s001
                                 + 0.961187 w_k1_DIS_s001
                                 + 1.14551 w_k1_FDX_s001
                                 + 0.696243 w_k1_IBM_s001
                                 + 0.834097 w_k1_INTC_s001
                                 + 0.620199 w_k1_KO_s001
                                 + 0.879567 w_k1_MAT_s001
                                 + 0.896912 w_k1_MSFT_s001
                                 + 0.986289 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.34645 w_k1_ADM_s001
                                 + 1.59239 w_k1_AXP_s001
                                 + 1.35715 w_k1_DIS_s001
                                 + 1.31356 w_k1_FDX_s001
                                 + 1.14353 w_k1_IBM_s001
                                 + 0.987398 w_k1_INTC_s001
                                 + 0.931208 w_k1_KO_s001 + 1.204 w_k1_MAT_s001
                                 + 1.15302 w_k1_MSFT_s001
                                 + 1.04531 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.09327 w_k1_ADM_s001
                                 + 0.981632 w_k1_AXP_s001
                                 + 0.908351 w_k1_DIS_s001
                                 + 0.927121 w_k1_FDX_s001
                                 + 1.12151 w_k1_IBM_s001
                                 + 0.700857 w_k1_INTC_s001
                                 + 0.821249 w_k1_KO_s001
                                 + 1.17802 w_k1_MAT_s001
                                 + 0.85761 w_k1_MSFT_s001
                                 + 0.977059 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.24421 w_k1_ADM_s001
                                 + 1.13318 w_k1_AXP_s001
                                 + 1.29829 w_k1_DIS_s001
                                 + 1.15989 w_k1_FDX_s001
                                 + 0.955685 w_k1_IBM_s001
                                 + 0.941244 w_k1_INTC_s001
                                 + 0.964833 w_k1_KO_s001
                                 + 0.703466 w_k1_MAT_s001
                                 + 1.08894 w_k1_MSFT_s001
                                 + 1.01511 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.631981 w_k1_ADM_s001
                                 + 0.703855 w_k1_AXP_s001
                                 + 0.557973 w_k1_DIS_s001
                                 + 0.704223 w_k1_FDX_s001
                                 + 0.88282 w_k1_IBM_s001
                                 + 0.860726 w_k1_INTC_s001
                                 + 0.692591 w_k1_KO_s001
                                 + 0.842449 w_k1_MAT_s001
                                 + 0.767121 w_k1_MSFT_s001
                                 + 0.801046 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.27566 w_k1_ADM_s002
                                 + 0.934044 w_k1_AXP_s002
                                 + 0.941027 w_k1_DIS_s002
                                 + 0.974035 w_k1_FDX_s002
                                 + 0.762616 w_k1_IBM_s002
                                 + 0.874645 w_k1_INTC_s002
                                 + 0.922265 w_k1_KO_s002
                                 + 0.786397 w_k1_MAT_s002
                                 + 1.05435 w_k1_MSFT_s002
                                 + 0.983049 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.819692 w_k1_ADM_s002
                                 + 0.994337 w_k1_AXP_s002
                                 + 0.790891 w_k1_DIS_s002
                                 + 0.800833 w_k1_FDX_s002
                                 + 1.15614 w_k1_IBM_s002
                                 + 0.840853 w_k1_INTC_s002
                                 + 0.96652 w_k1_KO_s002 + 1.17294 w_k1_MAT_s002
                                 + 0.756093 w_k1_MSFT_s002
                                 + 0.905481 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.81224 w_k1_ADM_s002
                                 + 1.53014 w_k1_AXP_s002
                                 + 1.45896 w_k1_DIS_s002
                                 + 1.13279 w_k1_FDX_s002
                                 + 1.64519 w_k1_IBM_s002
                                 + 1.01584 w_k1_INTC_s002
                                 + 1.37107 w_k1_KO_s002 + 1.68856 w_k1_MAT_s002
                                 + 1.16593 w_k1_MSFT_s002
                                 + 1.30324 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.3757 w_k1_ADM_s002
                                 + 1.31932 w_k1_AXP_s002
                                 + 1.38733 w_k1_DIS_s002
                                 + 1.22469 w_k1_FDX_s002
                                 + 0.965785 w_k1_IBM_s002
                                 + 0.925244 w_k1_INTC_s002
                                 + 1.18964 w_k1_KO_s002 + 1.24126 w_k1_MAT_s002
                                 + 1.01662 w_k1_MSFT_s002
                                 + 1.02836 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.03086 w_k1_ADM_s002
                                 + 0.909536 w_k1_AXP_s002
                                 + 0.698683 w_k1_DIS_s002
                                 + 0.826669 w_k1_FDX_s002
                                 + 0.610672 w_k1_IBM_s002
                                 + 0.766989 w_k1_INTC_s002
                                 + 0.651603 w_k1_KO_s002
                                 + 0.881824 w_k1_MAT_s002
                                 + 0.787145 w_k1_MSFT_s002
                                 + 0.840717 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.921582 w_k1_ADM_s002
                                 + 1.20533 w_k1_AXP_s002
                                 + 0.99474 w_k1_DIS_s002
                                 + 0.92979 w_k1_FDX_s002
                                 + 1.17309 w_k1_IBM_s002
                                 + 0.958121 w_k1_INTC_s002
                                 + 0.957361 w_k1_KO_s002
                                 + 1.24777 w_k1_MAT_s002
                                 + 0.941644 w_k1_MSFT_s002
                                 + 1.03373 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.53581 w_k1_ADM_s002
                                 + 1.45123 w_k1_AXP_s002
                                 + 1.24944 w_k1_DIS_s002
                                 + 1.22519 w_k1_FDX_s002 + 1.1461 w_k1_IBM_s002
                                 + 1.12462 w_k1_INTC_s002
                                 + 1.14846 w_k1_KO_s002 + 1.08076 w_k1_MAT_s002
                                 + 1.03538 w_k1_MSFT_s002
                                 + 1.18934 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.21733 w_k1_ADM_s002
                                 + 1.4144 w_k1_AXP_s002
                                 + 0.888658 w_k1_DIS_s002
                                 + 1.06404 w_k1_FDX_s002
                                 + 1.04283 w_k1_IBM_s002
                                 + 0.962021 w_k1_INTC_s002
                                 + 1.01727 w_k1_KO_s002 + 1.44058 w_k1_MAT_s002
                                 + 0.98018 w_k1_MSFT_s002
                                 + 1.06759 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.94034 w_k1_ADM_s003
                                 + 1.00093 w_k1_AXP_s003
                                 + 1.05209 w_k1_DIS_s003
                                 + 1.11941 w_k1_FDX_s003
                                 + 0.81884 w_k1_IBM_s003
                                 + 1.2631 w_k1_INTC_s003 + 1.04313 w_k1_KO_s003
                                 + 0.918257 w_k1_MAT_s003
                                 + 1.09432 w_k1_MSFT_s003
                                 + 0.986939 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.743015 w_k1_ADM_s003
                                 + 1.0218 w_k1_AXP_s003
                                 + 0.970733 w_k1_DIS_s003
                                 + 0.865804 w_k1_FDX_s003
                                 + 1.38208 w_k1_IBM_s003
                                 + 1.00485 w_k1_INTC_s003
                                 + 1.24161 w_k1_KO_s003 + 1.61369 w_k1_MAT_s003
                                 + 0.906312 w_k1_MSFT_s003
                                 + 1.0082 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.87386 w_k1_ADM_s003
                                 + 0.871553 w_k1_AXP_s003
                                 + 0.77113 w_k1_DIS_s003
                                 + 0.939216 w_k1_FDX_s003
                                 + 0.876774 w_k1_IBM_s003
                                 + 1.03822 w_k1_INTC_s003
                                 + 0.873928 w_k1_KO_s003
                                 + 0.911881 w_k1_MAT_s003
                                 + 0.86634 w_k1_MSFT_s003
                                 + 1.01548 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.14507 w_k1_ADM_s003
                                 + 1.33201 w_k1_AXP_s003
                                 + 1.72146 w_k1_DIS_s003
                                 + 1.34694 w_k1_FDX_s003
                                 + 1.24161 w_k1_IBM_s003
                                 + 1.3437 w_k1_INTC_s003 + 1.48235 w_k1_KO_s003
                                 + 1.43041 w_k1_MAT_s003
                                 + 1.13242 w_k1_MSFT_s003
                                 + 1.16844 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.17199 w_k1_ADM_s003
                                 + 1.15565 w_k1_AXP_s003
                                 + 1.39794 w_k1_DIS_s003
                                 + 1.21865 w_k1_FDX_s003
                                 + 0.986386 w_k1_IBM_s003
                                 + 1.02142 w_k1_INTC_s003
                                 + 1.27304 w_k1_KO_s003 + 1.03204 w_k1_MAT_s003
                                 + 1.05924 w_k1_MSFT_s003
                                 + 1.10208 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.804192 w_k1_ADM_s003
                                 + 0.987418 w_k1_AXP_s003
                                 + 0.962868 w_k1_DIS_s003
                                 + 0.952836 w_k1_FDX_s003
                                 + 1.12977 w_k1_IBM_s003
                                 + 0.879409 w_k1_INTC_s003
                                 + 1.12576 w_k1_KO_s003 + 1.3819 w_k1_MAT_s003
                                 + 0.830082 w_k1_MSFT_s003
                                 + 1.01301 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.677627 w_k1_ADM_s003
                                 + 0.689138 w_k1_AXP_s003
                                 + 0.595809 w_k1_DIS_s003
                                 + 0.715866 w_k1_FDX_s003
                                 + 0.84057 w_k1_IBM_s003
                                 + 0.707281 w_k1_INTC_s003
                                 + 0.810911 w_k1_KO_s003
                                 + 0.806717 w_k1_MAT_s003
                                 + 0.676211 w_k1_MSFT_s003
                                 + 0.863728 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.11196 w_k1_ADM_s003
                                 + 1.4292 w_k1_AXP_s003 + 1.1756 w_k1_DIS_s003
                                 + 1.1707 w_k1_FDX_s003 + 1.21918 w_k1_IBM_s003
                                 + 1.2534 w_k1_INTC_s003 + 1.31081 w_k1_KO_s003
                                 + 1.41704 w_k1_MAT_s003
                                 + 1.16925 w_k1_MSFT_s003
                                 + 1.17479 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.56895 w_k1_ADM_s004
                                 + 1.21468 w_k1_AXP_s004
                                 + 1.68701 w_k1_DIS_s004 + 1.2021 w_k1_FDX_s004
                                 + 1.59582 w_k1_IBM_s004
                                 + 1.01905 w_k1_INTC_s004
                                 + 1.27369 w_k1_KO_s004 + 1.09341 w_k1_MAT_s004
                                 + 1.19466 w_k1_MSFT_s004
                                 + 1.20903 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.01133 w_k1_ADM_s004
                                 + 0.979794 w_k1_AXP_s004
                                 + 0.705602 w_k1_DIS_s004
                                 + 0.901558 w_k1_FDX_s004
                                 + 0.735079 w_k1_IBM_s004
                                 + 0.822154 w_k1_INTC_s004
                                 + 0.700818 w_k1_KO_s004
                                 + 1.00312 w_k1_MAT_s004
                                 + 0.746451 w_k1_MSFT_s004
                                 + 0.926261 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.79797 w_k1_ADM_s004
                                 + 0.920098 w_k1_AXP_s004
                                 + 1.04566 w_k1_DIS_s004
                                 + 0.910021 w_k1_FDX_s004
                                 + 1.09114 w_k1_IBM_s004
                                 + 0.949367 w_k1_INTC_s004
                                 + 1.08274 w_k1_KO_s004
                                 + 0.777629 w_k1_MAT_s004
                                 + 0.893948 w_k1_MSFT_s004
                                 + 0.909885 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.26155 w_k1_ADM_s004
                                 + 0.897912 w_k1_AXP_s004
                                 + 0.968058 w_k1_DIS_s004
                                 + 1.04178 w_k1_FDX_s004
                                 + 0.721293 w_k1_IBM_s004
                                 + 0.710845 w_k1_INTC_s004
                                 + 0.709996 w_k1_KO_s004
                                 + 0.598627 w_k1_MAT_s004
                                 + 0.824163 w_k1_MSFT_s004
                                 + 0.909196 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.980031 w_k1_ADM_s004
                                 + 0.901354 w_k1_AXP_s004
                                 + 0.674327 w_k1_DIS_s004
                                 + 0.800437 w_k1_FDX_s004
                                 + 0.980601 w_k1_IBM_s004
                                 + 0.692299 w_k1_INTC_s004
                                 + 0.717682 w_k1_KO_s004
                                 + 0.804864 w_k1_MAT_s004
                                 + 0.701786 w_k1_MSFT_s004
                                 + 0.873838 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.05272 w_k1_ADM_s004
                                 + 1.28673 w_k1_AXP_s004
                                 + 1.24653 w_k1_DIS_s004
                                 + 1.02951 w_k1_FDX_s004
                                 + 1.37627 w_k1_IBM_s004
                                 + 0.986886 w_k1_INTC_s004
                                 + 1.17141 w_k1_KO_s004 + 1.12031 w_k1_MAT_s004
                                 + 1.00821 w_k1_MSFT_s004
                                 + 1.05957 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.830534 w_k1_ADM_s004
                                 + 0.592344 w_k1_AXP_s004
                                 + 0.666112 w_k1_DIS_s004
                                 + 0.71701 w_k1_FDX_s004
                                 + 0.974643 w_k1_IBM_s004
                                 + 0.761729 w_k1_INTC_s004
                                 + 0.79752 w_k1_KO_s004
                                 + 0.597499 w_k1_MAT_s004
                                 + 0.708274 w_k1_MSFT_s004
                                 + 0.787498 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.09083 w_k1_ADM_s004
                                 + 1.2345 w_k1_AXP_s004 + 1.43406 w_k1_DIS_s004
                                 + 1.1914 w_k1_FDX_s004
                                 + 0.986182 w_k1_IBM_s004
                                 + 1.10466 w_k1_INTC_s004
                                 + 1.12045 w_k1_KO_s004
                                 + 0.967599 w_k1_MAT_s004
                                 + 1.08712 w_k1_MSFT_s004
                                 + 1.12199 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.938413 w_k1_ADM_s005
                                 + 0.76466 w_k1_AXP_s005
                                 + 0.86771 w_k1_DIS_s005
                                 + 0.889661 w_k1_FDX_s005
                                 + 0.600934 w_k1_IBM_s005
                                 + 0.944973 w_k1_INTC_s005
                                 + 0.815272 w_k1_KO_s005
                                 + 0.57729 w_k1_MAT_s005
                                 + 0.886718 w_k1_MSFT_s005
                                 + 0.889245 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.50689 w_k1_ADM_s005
                                 + 1.4419 w_k1_AXP_s005 + 1.66487 w_k1_DIS_s005
                                 + 1.31485 w_k1_FDX_s005
                                 + 1.88734 w_k1_IBM_s005
                                 + 1.1835 w_k1_INTC_s005 + 1.75696 w_k1_KO_s005
                                 + 1.9143 w_k1_MAT_s005
                                 + 1.18973 w_k1_MSFT_s005
                                 + 1.44818 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.698757 w_k1_ADM_s005
                                 + 1.08264 w_k1_AXP_s005
                                 + 1.18049 w_k1_DIS_s005
                                 + 0.980838 w_k1_FDX_s005
                                 + 1.39215 w_k1_IBM_s005
                                 + 1.07423 w_k1_INTC_s005
                                 + 1.24621 w_k1_KO_s005 + 1.29694 w_k1_MAT_s005
                                 + 1.13808 w_k1_MSFT_s005
                                 + 1.09177 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.29113 w_k1_ADM_s005
                                 + 0.888159 w_k1_AXP_s005
                                 + 1.01782 w_k1_DIS_s005
                                 + 1.12043 w_k1_FDX_s005
                                 + 0.671552 w_k1_IBM_s005
                                 + 0.936219 w_k1_INTC_s005
                                 + 0.968081 w_k1_KO_s005
                                 + 0.625722 w_k1_MAT_s005
                                 + 1.12734 w_k1_MSFT_s005
                                 + 1.13732 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.744751 w_k1_ADM_s005
                                 + 0.692967 w_k1_AXP_s005
                                 + 0.58198 w_k1_DIS_s005
                                 + 0.692507 w_k1_FDX_s005
                                 + 0.731141 w_k1_IBM_s005
                                 + 0.652476 w_k1_INTC_s005
                                 + 0.640328 w_k1_KO_s005
                                 + 0.575934 w_k1_MAT_s005
                                 + 0.578948 w_k1_MSFT_s005
                                 + 0.686044 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.02187 w_k1_ADM_s005
                                 + 0.855921 w_k1_AXP_s005
                                 + 0.787664 w_k1_DIS_s005
                                 + 0.840815 w_k1_FDX_s005
                                 + 0.994467 w_k1_IBM_s005
                                 + 0.822541 w_k1_INTC_s005
                                 + 0.887113 w_k1_KO_s005
                                 + 0.868516 w_k1_MAT_s005
                                 + 0.825678 w_k1_MSFT_s005
                                 + 0.935133 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.4732 w_k1_ADM_s005
                                 + 1.57059 w_k1_AXP_s005 + 1.5473 w_k1_DIS_s005
                                 + 1.36321 w_k1_FDX_s005
                                 + 0.884017 w_k1_IBM_s005
                                 + 1.21491 w_k1_INTC_s005
                                 + 1.20071 w_k1_KO_s005 + 0.98876 w_k1_MAT_s005
                                 + 1.28964 w_k1_MSFT_s005
                                 + 1.21296 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.0356 w_k1_ADM_s005
                                 + 1.16626 w_k1_AXP_s005
                                 + 1.14497 w_k1_DIS_s005
                                 + 1.02328 w_k1_FDX_s005
                                 + 0.993613 w_k1_IBM_s005
                                 + 1.06981 w_k1_INTC_s005
                                 + 0.946109 w_k1_KO_s005
                                 + 0.946587 w_k1_MAT_s005
                                 + 1.03989 w_k1_MSFT_s005
                                 + 1.04485 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 2.0416 w_k1_ADM_s006
                                 + 1.94535 w_k1_AXP_s006
                                 + 1.61288 w_k1_DIS_s006
                                 + 1.47209 w_k1_FDX_s006
                                 + 1.40238 w_k1_IBM_s006
                                 + 1.21208 w_k1_INTC_s006 + 1.5723 w_k1_KO_s006
                                 + 1.73017 w_k1_MAT_s006
                                 + 1.11647 w_k1_MSFT_s006
                                 + 1.42242 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.25959 w_k1_ADM_s006
                                 + 1.11597 w_k1_AXP_s006
                                 + 1.23545 w_k1_DIS_s006
                                 + 1.01027 w_k1_FDX_s006
                                 + 1.39297 w_k1_IBM_s006
                                 + 1.2495 w_k1_INTC_s006 + 1.4592 w_k1_KO_s006
                                 + 1.40749 w_k1_MAT_s006
                                 + 1.28443 w_k1_MSFT_s006
                                 + 1.28685 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.921369 w_k1_ADM_s006
                                 + 0.807766 w_k1_AXP_s006
                                 + 0.851242 w_k1_DIS_s006
                                 + 0.815277 w_k1_FDX_s006
                                 + 1.22391 w_k1_IBM_s006
                                 + 1.06708 w_k1_INTC_s006
                                 + 1.01473 w_k1_KO_s006
                                 + 0.959023 w_k1_MAT_s006
                                 + 0.903666 w_k1_MSFT_s006
                                 + 0.934661 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.2493 w_k1_ADM_s006
                                 + 1.2072 w_k1_AXP_s006 + 1.22009 w_k1_DIS_s006
                                 + 1.21629 w_k1_FDX_s006
                                 + 0.983416 w_k1_IBM_s006
                                 + 1.22291 w_k1_INTC_s006
                                 + 1.17735 w_k1_KO_s006
                                 + 0.942412 w_k1_MAT_s006
                                 + 1.12355 w_k1_MSFT_s006
                                 + 1.13288 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.00776 w_k1_ADM_s006
                                 + 0.592905 w_k1_AXP_s006
                                 + 0.565533 w_k1_DIS_s006
                                 + 0.716369 w_k1_FDX_s006
                                 + 0.893892 w_k1_IBM_s006
                                 + 0.801157 w_k1_INTC_s006
                                 + 0.785899 w_k1_KO_s006
                                 + 0.617083 w_k1_MAT_s006
                                 + 0.716845 w_k1_MSFT_s006
                                 + 0.968847 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.3077 w_k1_ADM_s006
                                 + 0.953821 w_k1_AXP_s006
                                 + 0.848938 w_k1_DIS_s006
                                 + 1.07083 w_k1_FDX_s006
                                 + 0.66924 w_k1_IBM_s006
                                 + 0.769763 w_k1_INTC_s006
                                 + 0.88482 w_k1_KO_s006
                                 + 0.740654 w_k1_MAT_s006
                                 + 0.963717 w_k1_MSFT_s006
                                 + 1.02241 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.832222 w_k1_ADM_s006
                                 + 0.863711 w_k1_AXP_s006
                                 + 0.872876 w_k1_DIS_s006
                                 + 0.890863 w_k1_FDX_s006
                                 + 0.972251 w_k1_IBM_s006
                                 + 1.10033 w_k1_INTC_s006
                                 + 1.10724 w_k1_KO_s006 + 0.92862 w_k1_MAT_s006
                                 + 0.826314 w_k1_MSFT_s006
                                 + 1.02965 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.942941 w_k1_ADM_s006
                                 + 0.924949 w_k1_AXP_s006
                                 + 0.737525 w_k1_DIS_s006
                                 + 0.850174 w_k1_FDX_s006
                                 + 1.07944 w_k1_IBM_s006
                                 + 1.02065 w_k1_INTC_s006
                                 + 0.906088 w_k1_KO_s006
                                 + 0.824178 w_k1_MAT_s006
                                 + 0.797292 w_k1_MSFT_s006
                                 + 0.996721 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.885815 w_k1_ADM_s007
                                 + 0.865724 w_k1_AXP_s007
                                 + 0.849152 w_k1_DIS_s007
                                 + 0.863588 w_k1_FDX_s007
                                 + 0.856978 w_k1_IBM_s007
                                 + 0.896659 w_k1_INTC_s007
                                 + 0.959165 w_k1_KO_s007
                                 + 1.35309 w_k1_MAT_s007
                                 + 0.94362 w_k1_MSFT_s007
                                 + 0.980468 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.687902 w_k1_ADM_s007
                                 + 0.61387 w_k1_AXP_s007
                                 + 0.544061 w_k1_DIS_s007
                                 + 0.694894 w_k1_FDX_s007
                                 + 0.693068 w_k1_IBM_s007
                                 + 0.785931 w_k1_INTC_s007
                                 + 0.710215 w_k1_KO_s007
                                 + 0.966514 w_k1_MAT_s007
                                 + 0.697741 w_k1_MSFT_s007
                                 + 0.804804 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.26945 w_k1_ADM_s007
                                 + 1.37365 w_k1_AXP_s007
                                 + 1.10494 w_k1_DIS_s007 + 1.2038 w_k1_FDX_s007
                                 + 0.909342 w_k1_IBM_s007
                                 + 1.2733 w_k1_INTC_s007 + 1.00959 w_k1_KO_s007
                                 + 0.98673 w_k1_MAT_s007
                                 + 1.16004 w_k1_MSFT_s007
                                 + 1.15603 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.16794 w_k1_ADM_s007
                                 + 1.33201 w_k1_AXP_s007 + 1.6848 w_k1_DIS_s007
                                 + 1.40916 w_k1_FDX_s007 + 1.1719 w_k1_IBM_s007
                                 + 1.46639 w_k1_INTC_s007
                                 + 1.43651 w_k1_KO_s007 + 1.36617 w_k1_MAT_s007
                                 + 1.4452 w_k1_MSFT_s007 + 1.13324 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.881891 w_k1_ADM_s007
                                 + 0.836685 w_k1_AXP_s007
                                 + 0.80051 w_k1_DIS_s007
                                 + 0.838171 w_k1_FDX_s007
                                 + 1.09478 w_k1_IBM_s007
                                 + 0.830149 w_k1_INTC_s007
                                 + 0.968216 w_k1_KO_s007
                                 + 1.54057 w_k1_MAT_s007
                                 + 0.914401 w_k1_MSFT_s007
                                 + 0.956351 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.948433 w_k1_ADM_s007
                                 + 0.994483 w_k1_AXP_s007
                                 + 0.942524 w_k1_DIS_s007
                                 + 0.985242 w_k1_FDX_s007
                                 + 0.838516 w_k1_IBM_s007
                                 + 1.07748 w_k1_INTC_s007
                                 + 0.847334 w_k1_KO_s007
                                 + 0.954384 w_k1_MAT_s007
                                 + 1.08713 w_k1_MSFT_s007
                                 + 0.993125 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.837792 w_k1_ADM_s007
                                 + 0.892123 w_k1_AXP_s007
                                 + 0.57531 w_k1_DIS_s007
                                 + 0.810738 w_k1_FDX_s007
                                 + 0.755183 w_k1_IBM_s007
                                 + 0.875528 w_k1_INTC_s007
                                 + 0.872155 w_k1_KO_s007
                                 + 0.870239 w_k1_MAT_s007
                                 + 0.83536 w_k1_MSFT_s007
                                 + 0.88209 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.16922 w_k1_ADM_s007
                                 + 1.15405 w_k1_AXP_s007
                                 + 1.17117 w_k1_DIS_s007
                                 + 1.04916 w_k1_FDX_s007
                                 + 1.47793 w_k1_IBM_s007
                                 + 1.09309 w_k1_INTC_s007
                                 + 1.43292 w_k1_KO_s007 + 1.55286 w_k1_MAT_s007
                                 + 1.14496 w_k1_MSFT_s007
                                 + 1.00817 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.889356 w_k1_ADM_s010
                                 + 0.893492 w_k1_AXP_s010
                                 + 0.760452 w_k1_DIS_s010
                                 + 0.904905 w_k1_FDX_s010
                                 + 1.17915 w_k1_IBM_s010
                                 + 1.07081 w_k1_INTC_s010
                                 + 1.07166 w_k1_KO_s010 + 1.082 w_k1_MAT_s010
                                 + 0.983708 w_k1_MSFT_s010
                                 + 1.05916 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.738086 w_k1_ADM_s010
                                 + 0.624532 w_k1_AXP_s010
                                 + 0.700388 w_k1_DIS_s010
                                 + 0.687125 w_k1_FDX_s010
                                 + 0.864695 w_k1_IBM_s010
                                 + 0.703571 w_k1_INTC_s010
                                 + 0.844005 w_k1_KO_s010
                                 + 0.662484 w_k1_MAT_s010
                                 + 0.791321 w_k1_MSFT_s010
                                 + 0.8059 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.03429 w_k1_ADM_s010
                                 + 0.84981 w_k1_AXP_s010
                                 + 1.25585 w_k1_DIS_s010
                                 + 1.16632 w_k1_FDX_s010
                                 + 0.99095 w_k1_IBM_s010
                                 + 1.22666 w_k1_INTC_s010
                                 + 1.29068 w_k1_KO_s010 + 1.05284 w_k1_MAT_s010
                                 + 1.43679 w_k1_MSFT_s010
                                 + 1.21782 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.986261 w_k1_ADM_s010
                                 + 0.704552 w_k1_AXP_s010
                                 + 0.999026 w_k1_DIS_s010
                                 + 0.892468 w_k1_FDX_s010
                                 + 0.771898 w_k1_IBM_s010
                                 + 0.788447 w_k1_INTC_s010
                                 + 0.911538 w_k1_KO_s010
                                 + 0.545122 w_k1_MAT_s010
                                 + 0.994738 w_k1_MSFT_s010
                                 + 0.978235 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.01658 w_k1_ADM_s010
                                 + 1.43265 w_k1_AXP_s010
                                 + 1.25464 w_k1_DIS_s010
                                 + 1.20515 w_k1_FDX_s010
                                 + 1.22734 w_k1_IBM_s010
                                 + 1.15957 w_k1_INTC_s010
                                 + 1.15719 w_k1_KO_s010 + 1.22389 w_k1_MAT_s010
                                 + 1.11328 w_k1_MSFT_s010
                                 + 1.11025 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.34224 w_k1_ADM_s010
                                 + 1.75851 w_k1_AXP_s010
                                 + 1.67809 w_k1_DIS_s010 + 1.3315 w_k1_FDX_s010
                                 + 1.69382 w_k1_IBM_s010
                                 + 1.16076 w_k1_INTC_s010
                                 + 1.50248 w_k1_KO_s010 + 2.26616 w_k1_MAT_s010
                                 + 1.34356 w_k1_MSFT_s010
                                 + 1.34242 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.0501 w_k1_ADM_s010
                                 + 0.7333 w_k1_AXP_s010
                                 + 0.861443 w_k1_DIS_s010
                                 + 0.987988 w_k1_FDX_s010
                                 + 0.550447 w_k1_IBM_s010
                                 + 0.936511 w_k1_INTC_s010
                                 + 0.783322 w_k1_KO_s010
                                 + 0.632766 w_k1_MAT_s010
                                 + 1.22011 w_k1_MSFT_s010
                                 + 0.996214 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.854132 w_k1_ADM_s010
                                 + 0.573253 w_k1_AXP_s010
                                 + 0.70702 w_k1_DIS_s010
                                 + 0.754917 w_k1_FDX_s010
                                 + 0.484583 w_k1_IBM_s010
                                 + 0.733426 w_k1_INTC_s010
                                 + 0.711028 w_k1_KO_s010
                                 + 0.48903 w_k1_MAT_s010
                                 + 0.843512 w_k1_MSFT_s010
                                 + 0.777915 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.08454 w_k1_ADM_s011
                                 + 1.20176 w_k1_AXP_s011
                                 + 1.34916 w_k1_DIS_s011
                                 + 0.967536 w_k1_FDX_s011
                                 + 1.59544 w_k1_IBM_s011
                                 + 1.22294 w_k1_INTC_s011
                                 + 1.46445 w_k1_KO_s011 + 1.24571 w_k1_MAT_s011
                                 + 1.18793 w_k1_MSFT_s011 + 1.0381 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.916671 w_k1_ADM_s011
                                 + 0.758833 w_k1_AXP_s011
                                 + 0.870591 w_k1_DIS_s011
                                 + 0.8301 w_k1_FDX_s011
                                 + 0.922054 w_k1_IBM_s011
                                 + 0.816591 w_k1_INTC_s011
                                 + 0.958504 w_k1_KO_s011
                                 + 0.777101 w_k1_MAT_s011
                                 + 0.86615 w_k1_MSFT_s011
                                 + 0.858798 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.28845 w_k1_ADM_s011
                                 + 1.37845 w_k1_AXP_s011
                                 + 1.09933 w_k1_DIS_s011
                                 + 1.11581 w_k1_FDX_s011
                                 + 1.22703 w_k1_IBM_s011
                                 + 1.13879 w_k1_INTC_s011
                                 + 1.11885 w_k1_KO_s011 + 1.15269 w_k1_MAT_s011
                                 + 1.0901 w_k1_MSFT_s011
                                 + 0.961474 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.05955 w_k1_ADM_s011
                                 + 0.922755 w_k1_AXP_s011
                                 + 0.957189 w_k1_DIS_s011
                                 + 0.917854 w_k1_FDX_s011
                                 + 1.28267 w_k1_IBM_s011
                                 + 0.840583 w_k1_INTC_s011
                                 + 1.18613 w_k1_KO_s011 + 1.28664 w_k1_MAT_s011
                                 + 0.963201 w_k1_MSFT_s011
                                 + 0.973102 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.11474 w_k1_ADM_s011
                                 + 0.909456 w_k1_AXP_s011
                                 + 0.920147 w_k1_DIS_s011
                                 + 1.0216 w_k1_FDX_s011
                                 + 0.795894 w_k1_IBM_s011
                                 + 0.974618 w_k1_INTC_s011
                                 + 0.869382 w_k1_KO_s011
                                 + 0.763293 w_k1_MAT_s011
                                 + 0.974132 w_k1_MSFT_s011
                                 + 0.866987 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01368 w_k1_ADM_s011
                                 + 1.25449 w_k1_AXP_s011
                                 + 1.57091 w_k1_DIS_s011
                                 + 1.28108 w_k1_FDX_s011 + 1.178 w_k1_IBM_s011
                                 + 1.38954 w_k1_INTC_s011
                                 + 1.33842 w_k1_KO_s011 + 1.42608 w_k1_MAT_s011
                                 + 1.2634 w_k1_MSFT_s011 + 1.23108 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.46044 w_k1_ADM_s011
                                 + 1.03796 w_k1_AXP_s011
                                 + 1.11377 w_k1_DIS_s011
                                 + 1.23171 w_k1_FDX_s011
                                 + 0.908112 w_k1_IBM_s011
                                 + 1.01072 w_k1_INTC_s011
                                 + 0.919756 w_k1_KO_s011
                                 + 0.573106 w_k1_MAT_s011
                                 + 1.1122 w_k1_MSFT_s011
                                 + 0.861696 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.971038 w_k1_ADM_s011
                                 + 0.619814 w_k1_AXP_s011
                                 + 0.710123 w_k1_DIS_s011
                                 + 0.799612 w_k1_FDX_s011
                                 + 0.580512 w_k1_IBM_s011
                                 + 0.75826 w_k1_INTC_s011
                                 + 0.729169 w_k1_KO_s011
                                 + 0.702825 w_k1_MAT_s011
                                 + 0.830317 w_k1_MSFT_s011
                                 + 0.818385 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.927386 w_k1_ADM_s012
                                 + 0.875415 w_k1_AXP_s012
                                 + 0.838435 w_k1_DIS_s012
                                 + 0.919613 w_k1_FDX_s012
                                 + 0.705822 w_k1_IBM_s012
                                 + 0.815193 w_k1_INTC_s012
                                 + 0.758527 w_k1_KO_s012
                                 + 0.808807 w_k1_MAT_s012
                                 + 0.971029 w_k1_MSFT_s012
                                 + 0.859931 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.847565 w_k1_ADM_s012
                                 + 0.674303 w_k1_AXP_s012
                                 + 0.726667 w_k1_DIS_s012
                                 + 0.794072 w_k1_FDX_s012
                                 + 0.78732 w_k1_IBM_s012
                                 + 0.769718 w_k1_INTC_s012
                                 + 0.846067 w_k1_KO_s012
                                 + 0.588705 w_k1_MAT_s012
                                 + 0.806204 w_k1_MSFT_s012
                                 + 0.830117 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.16813 w_k1_ADM_s012
                                 + 1.33905 w_k1_AXP_s012
                                 + 1.45439 w_k1_DIS_s012 + 1.2327 w_k1_FDX_s012
                                 + 1.77658 w_k1_IBM_s012
                                 + 1.34536 w_k1_INTC_s012
                                 + 1.47305 w_k1_KO_s012 + 1.73958 w_k1_MAT_s012
                                 + 1.27648 w_k1_MSFT_s012
                                 + 1.15879 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.40463 w_k1_ADM_s012
                                 + 1.30654 w_k1_AXP_s012
                                 + 1.53348 w_k1_DIS_s012
                                 + 1.28988 w_k1_FDX_s012
                                 + 1.03244 w_k1_IBM_s012
                                 + 1.3557 w_k1_INTC_s012 + 1.50259 w_k1_KO_s012
                                 + 1.16266 w_k1_MAT_s012
                                 + 1.45828 w_k1_MSFT_s012
                                 + 1.27926 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.0506 w_k1_ADM_s012
                                 + 0.987658 w_k1_AXP_s012
                                 + 0.965278 w_k1_DIS_s012
                                 + 1.00812 w_k1_FDX_s012
                                 + 0.785049 w_k1_IBM_s012
                                 + 1.05689 w_k1_INTC_s012
                                 + 0.974302 w_k1_KO_s012
                                 + 0.852715 w_k1_MAT_s012
                                 + 1.04141 w_k1_MSFT_s012
                                 + 0.95142 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.15069 w_k1_ADM_s012
                                 + 1.24164 w_k1_AXP_s012 + 1.2686 w_k1_DIS_s012
                                 + 1.12517 w_k1_FDX_s012
                                 + 1.02328 w_k1_IBM_s012
                                 + 1.05728 w_k1_INTC_s012
                                 + 1.16766 w_k1_KO_s012 + 1.06504 w_k1_MAT_s012
                                 + 1.03391 w_k1_MSFT_s012
                                 + 1.15896 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.872656 w_k1_ADM_s012
                                 + 0.840293 w_k1_AXP_s012
                                 + 1.22192 w_k1_DIS_s012
                                 + 1.06411 w_k1_FDX_s012
                                 + 0.864165 w_k1_IBM_s012
                                 + 1.14132 w_k1_INTC_s012
                                 + 1.07033 w_k1_KO_s012
                                 + 0.839141 w_k1_MAT_s012
                                 + 1.27451 w_k1_MSFT_s012
                                 + 1.04642 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.839692 w_k1_ADM_s012
                                 + 1.18204 w_k1_AXP_s012
                                 + 0.822776 w_k1_DIS_s012
                                 + 0.887893 w_k1_FDX_s012
                                 + 1.15573 w_k1_IBM_s012
                                 + 1.10595 w_k1_INTC_s012
                                 + 1.14225 w_k1_KO_s012 + 1.1152 w_k1_MAT_s012
                                 + 0.994806 w_k1_MSFT_s012
                                 + 1.00644 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.35074 w_k1_ADM_s013
                                 + 1.31591 w_k1_AXP_s013
                                 + 1.47187 w_k1_DIS_s013
                                 + 1.24935 w_k1_FDX_s013
                                 + 1.18871 w_k1_IBM_s013
                                 + 1.09683 w_k1_INTC_s013
                                 + 1.14177 w_k1_KO_s013 + 1.15167 w_k1_MAT_s013
                                 + 1.24805 w_k1_MSFT_s013
                                 + 1.00719 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.800747 w_k1_ADM_s013
                                 + 0.853777 w_k1_AXP_s013
                                 + 0.865716 w_k1_DIS_s013
                                 + 0.852994 w_k1_FDX_s013
                                 + 0.822125 w_k1_IBM_s013
                                 + 1.04039 w_k1_INTC_s013
                                 + 0.864748 w_k1_KO_s013
                                 + 0.925991 w_k1_MAT_s013
                                 + 0.884858 w_k1_MSFT_s013
                                 + 0.896764 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.884923 w_k1_ADM_s013
                                 + 1.05284 w_k1_AXP_s013
                                 + 0.874212 w_k1_DIS_s013
                                 + 1.07648 w_k1_FDX_s013
                                 + 0.783211 w_k1_IBM_s013
                                 + 1.08416 w_k1_INTC_s013
                                 + 0.861978 w_k1_KO_s013
                                 + 0.813662 w_k1_MAT_s013
                                 + 1.02626 w_k1_MSFT_s013
                                 + 0.949126 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.81586 w_k1_ADM_s013
                                 + 0.604212 w_k1_AXP_s013
                                 + 0.831583 w_k1_DIS_s013
                                 + 0.845861 w_k1_FDX_s013 + 0.857 w_k1_IBM_s013
                                 + 0.751694 w_k1_INTC_s013
                                 + 0.838307 w_k1_KO_s013
                                 + 0.647181 w_k1_MAT_s013
                                 + 0.824381 w_k1_MSFT_s013
                                 + 0.845004 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.3029 w_k1_ADM_s013
                                 + 1.63949 w_k1_AXP_s013
                                 + 1.07092 w_k1_DIS_s013
                                 + 1.15927 w_k1_FDX_s013
                                 + 1.12818 w_k1_IBM_s013
                                 + 0.974283 w_k1_INTC_s013
                                 + 1.05823 w_k1_KO_s013 + 1.61781 w_k1_MAT_s013
                                 + 0.898218 w_k1_MSFT_s013
                                 + 1.09166 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.839594 w_k1_ADM_s013
                                 + 1.14161 w_k1_AXP_s013
                                 + 1.20929 w_k1_DIS_s013
                                 + 1.01263 w_k1_FDX_s013 + 1.6371 w_k1_IBM_s013
                                 + 0.718854 w_k1_INTC_s013
                                 + 1.27112 w_k1_KO_s013 + 2.21532 w_k1_MAT_s013
                                 + 0.822833 w_k1_MSFT_s013 + 1.094 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.28417 w_k1_ADM_s013
                                 + 1.04538 w_k1_AXP_s013
                                 + 1.20641 w_k1_DIS_s013 + 1.2483 w_k1_FDX_s013
                                 + 0.792014 w_k1_IBM_s013
                                 + 1.03371 w_k1_INTC_s013
                                 + 1.00996 w_k1_KO_s013 + 1.14116 w_k1_MAT_s013
                                 + 1.14315 w_k1_MSFT_s013
                                 + 0.987037 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.922449 w_k1_ADM_s013
                                 + 0.910082 w_k1_AXP_s013
                                 + 0.712297 w_k1_DIS_s013
                                 + 0.816374 w_k1_FDX_s013
                                 + 1.18437 w_k1_IBM_s013
                                 + 0.989376 w_k1_INTC_s013
                                 + 0.942869 w_k1_KO_s013
                                 + 1.27991 w_k1_MAT_s013
                                 + 0.796937 w_k1_MSFT_s013
                                 + 0.974842 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.03216 w_k1_ADM_s014
                                 + 1.15234 w_k1_AXP_s014
                                 + 0.880152 w_k1_DIS_s014
                                 + 0.901688 w_k1_FDX_s014
                                 + 0.916119 w_k1_IBM_s014
                                 + 1.06138 w_k1_INTC_s014
                                 + 0.862651 w_k1_KO_s014
                                 + 1.06709 w_k1_MAT_s014
                                 + 0.853063 w_k1_MSFT_s014
                                 + 0.933004 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.15449 w_k1_ADM_s014
                                 + 1.29968 w_k1_AXP_s014
                                 + 1.52412 w_k1_DIS_s014
                                 + 1.33688 w_k1_FDX_s014
                                 + 1.42147 w_k1_IBM_s014
                                 + 1.50228 w_k1_INTC_s014
                                 + 1.34698 w_k1_KO_s014 + 1.27869 w_k1_MAT_s014
                                 + 1.37264 w_k1_MSFT_s014
                                 + 1.02521 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.748306 w_k1_ADM_s014
                                 + 0.836853 w_k1_AXP_s014
                                 + 0.846527 w_k1_DIS_s014
                                 + 0.907522 w_k1_FDX_s014
                                 + 1.10944 w_k1_IBM_s014
                                 + 1.18891 w_k1_INTC_s014
                                 + 1.11585 w_k1_KO_s014
                                 + 0.854877 w_k1_MAT_s014
                                 + 0.992344 w_k1_MSFT_s014
                                 + 0.925988 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.905265 w_k1_ADM_s014
                                 + 0.943821 w_k1_AXP_s014
                                 + 1.18113 w_k1_DIS_s014
                                 + 1.17477 w_k1_FDX_s014
                                 + 0.882682 w_k1_IBM_s014
                                 + 1.00564 w_k1_INTC_s014
                                 + 0.949668 w_k1_KO_s014
                                 + 0.886421 w_k1_MAT_s014
                                 + 1.17382 w_k1_MSFT_s014
                                 + 0.932386 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.19379 w_k1_ADM_s014
                                 + 1.20285 w_k1_AXP_s014
                                 + 1.18355 w_k1_DIS_s014
                                 + 1.14606 w_k1_FDX_s014
                                 + 1.15168 w_k1_IBM_s014
                                 + 1.40265 w_k1_INTC_s014
                                 + 1.15943 w_k1_KO_s014
                                 + 0.778864 w_k1_MAT_s014
                                 + 1.21526 w_k1_MSFT_s014
                                 + 0.991504 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.893285 w_k1_ADM_s014
                                 + 0.609507 w_k1_AXP_s014
                                 + 0.691558 w_k1_DIS_s014
                                 + 0.85268 w_k1_FDX_s014
                                 + 0.732279 w_k1_IBM_s014
                                 + 1.07901 w_k1_INTC_s014
                                 + 0.808334 w_k1_KO_s014
                                 + 0.389604 w_k1_MAT_s014
                                 + 0.880986 w_k1_MSFT_s014
                                 + 0.786229 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.789411 w_k1_ADM_s014
                                 + 1.04286 w_k1_AXP_s014
                                 + 1.03268 w_k1_DIS_s014
                                 + 0.859608 w_k1_FDX_s014
                                 + 1.70198 w_k1_IBM_s014
                                 + 1.14135 w_k1_INTC_s014
                                 + 1.23396 w_k1_KO_s014 + 1.40828 w_k1_MAT_s014
                                 + 0.966948 w_k1_MSFT_s014
                                 + 0.900409 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.14263 w_k1_ADM_s014
                                 + 1.24364 w_k1_AXP_s014
                                 + 1.16722 w_k1_DIS_s014
                                 + 1.01615 w_k1_FDX_s014 + 1.3926 w_k1_IBM_s014
                                 + 1.18448 w_k1_INTC_s014
                                 + 1.13673 w_k1_KO_s014
                                 + 0.929527 w_k1_MAT_s014
                                 + 0.964424 w_k1_MSFT_s014
                                 + 0.977987 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.843786 w_k1_ADM_s015
                                 + 0.829971 w_k1_AXP_s015
                                 + 0.859066 w_k1_DIS_s015
                                 + 0.929596 w_k1_FDX_s015
                                 + 0.760179 w_k1_IBM_s015
                                 + 1.06498 w_k1_INTC_s015
                                 + 0.730973 w_k1_KO_s015
                                 + 0.710051 w_k1_MAT_s015
                                 + 1.01324 w_k1_MSFT_s015
                                 + 0.883597 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.68753 w_k1_ADM_s015
                                 + 0.725019 w_k1_AXP_s015
                                 + 0.770355 w_k1_DIS_s015
                                 + 0.741114 w_k1_FDX_s015
                                 + 0.929329 w_k1_IBM_s015
                                 + 0.876713 w_k1_INTC_s015
                                 + 0.751463 w_k1_KO_s015
                                 + 0.765801 w_k1_MAT_s015
                                 + 0.833791 w_k1_MSFT_s015
                                 + 0.916282 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.0253 w_k1_ADM_s015
                                 + 0.963317 w_k1_AXP_s015
                                 + 0.750372 w_k1_DIS_s015
                                 + 0.84182 w_k1_FDX_s015
                                 + 1.20877 w_k1_IBM_s015
                                 + 1.06955 w_k1_INTC_s015
                                 + 0.888505 w_k1_KO_s015
                                 + 1.10048 w_k1_MAT_s015
                                 + 0.975906 w_k1_MSFT_s015
                                 + 0.895786 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.986808 w_k1_ADM_s015
                                 + 1.1615 w_k1_AXP_s015 + 1.29543 w_k1_DIS_s015
                                 + 1.0903 w_k1_FDX_s015 + 1.39486 w_k1_IBM_s015
                                 + 1.71954 w_k1_INTC_s015
                                 + 1.08906 w_k1_KO_s015
                                 + 0.790114 w_k1_MAT_s015
                                 + 1.20997 w_k1_MSFT_s015
                                 + 1.02103 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.785236 w_k1_ADM_s015
                                 + 0.927617 w_k1_AXP_s015
                                 + 0.872065 w_k1_DIS_s015
                                 + 1.00142 w_k1_FDX_s015
                                 + 0.822864 w_k1_IBM_s015
                                 + 1.31161 w_k1_INTC_s015
                                 + 0.830006 w_k1_KO_s015
                                 + 0.542833 w_k1_MAT_s015
                                 + 0.997675 w_k1_MSFT_s015
                                 + 0.90914 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.04455 w_k1_ADM_s015
                                 + 1.06946 w_k1_AXP_s015
                                 + 1.05613 w_k1_DIS_s015
                                 + 1.02006 w_k1_FDX_s015
                                 + 1.00436 w_k1_IBM_s015
                                 + 0.914321 w_k1_INTC_s015
                                 + 1.01458 w_k1_KO_s015 + 1.14139 w_k1_MAT_s015
                                 + 1.13861 w_k1_MSFT_s015
                                 + 1.01435 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.19379 w_k1_ADM_s015
                                 + 1.38108 w_k1_AXP_s015
                                 + 1.43738 w_k1_DIS_s015
                                 + 1.32645 w_k1_FDX_s015
                                 + 1.08439 w_k1_IBM_s015
                                 + 1.21563 w_k1_INTC_s015
                                 + 1.12496 w_k1_KO_s015 + 1.03636 w_k1_MAT_s015
                                 + 1.24944 w_k1_MSFT_s015
                                 + 1.15432 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.937271 w_k1_ADM_s015
                                 + 0.646933 w_k1_AXP_s015
                                 + 0.714908 w_k1_DIS_s015
                                 + 0.808404 w_k1_FDX_s015
                                 + 0.600505 w_k1_IBM_s015
                                 + 0.866829 w_k1_INTC_s015
                                 + 0.642749 w_k1_KO_s015
                                 + 0.486548 w_k1_MAT_s015
                                 + 0.963646 w_k1_MSFT_s015
                                 + 0.870945 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.71231 w_k1_ADM_s016
                                 + 0.868235 w_k1_AXP_s016
                                 + 0.679614 w_k1_DIS_s016
                                 + 0.789684 w_k1_FDX_s016
                                 + 0.788532 w_k1_IBM_s016
                                 + 0.979081 w_k1_INTC_s016
                                 + 0.817679 w_k1_KO_s016
                                 + 0.721765 w_k1_MAT_s016
                                 + 0.958098 w_k1_MSFT_s016
                                 + 0.843991 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.0885 w_k1_ADM_s016
                                 + 0.948099 w_k1_AXP_s016
                                 + 0.962212 w_k1_DIS_s016
                                 + 0.990232 w_k1_FDX_s016
                                 + 0.821662 w_k1_IBM_s016
                                 + 1.22719 w_k1_INTC_s016 + 0.8633 w_k1_KO_s016
                                 + 0.761551 w_k1_MAT_s016
                                 + 1.32841 w_k1_MSFT_s016
                                 + 0.937047 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.914808 w_k1_ADM_s016
                                 + 1.04624 w_k1_AXP_s016
                                 + 1.13796 w_k1_DIS_s016
                                 + 1.03284 w_k1_FDX_s016
                                 + 1.08744 w_k1_IBM_s016
                                 + 1.23826 w_k1_INTC_s016
                                 + 1.10418 w_k1_KO_s016 + 0.75808 w_k1_MAT_s016
                                 + 1.13779 w_k1_MSFT_s016
                                 + 0.995285 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.901444 w_k1_ADM_s016
                                 + 1.15528 w_k1_AXP_s016
                                 + 0.911702 w_k1_DIS_s016
                                 + 0.911475 w_k1_FDX_s016
                                 + 1.38264 w_k1_IBM_s016
                                 + 1.18663 w_k1_INTC_s016 + 1.1373 w_k1_KO_s016
                                 + 1.95943 w_k1_MAT_s016
                                 + 1.08547 w_k1_MSFT_s016
                                 + 0.905044 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.28001 w_k1_ADM_s016
                                 + 1.29689 w_k1_AXP_s016 + 1.2484 w_k1_DIS_s016
                                 + 1.22307 w_k1_FDX_s016
                                 + 0.862434 w_k1_IBM_s016
                                 + 1.29653 w_k1_INTC_s016
                                 + 1.05683 w_k1_KO_s016
                                 + 0.687875 w_k1_MAT_s016
                                 + 1.33283 w_k1_MSFT_s016
                                 + 1.02491 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.846392 w_k1_ADM_s016
                                 + 0.905577 w_k1_AXP_s016
                                 + 0.760509 w_k1_DIS_s016
                                 + 1.0016 w_k1_FDX_s016
                                 + 0.544136 w_k1_IBM_s016
                                 + 1.24098 w_k1_INTC_s016
                                 + 0.772551 w_k1_KO_s016
                                 + 0.688022 w_k1_MAT_s016
                                 + 1.12542 w_k1_MSFT_s016
                                 + 0.894311 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.922475 w_k1_ADM_s016
                                 + 0.96374 w_k1_AXP_s016
                                 + 0.787364 w_k1_DIS_s016
                                 + 0.854556 w_k1_FDX_s016
                                 + 0.980672 w_k1_IBM_s016
                                 + 1.03921 w_k1_INTC_s016
                                 + 0.95572 w_k1_KO_s016
                                 + 0.907912 w_k1_MAT_s016
                                 + 1.01369 w_k1_MSFT_s016
                                 + 0.859324 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01629 w_k1_ADM_s016
                                 + 1.6171 w_k1_AXP_s016 + 1.47701 w_k1_DIS_s016
                                 + 1.14276 w_k1_FDX_s016
                                 + 1.48324 w_k1_IBM_s016
                                 + 1.35842 w_k1_INTC_s016
                                 + 1.32968 w_k1_KO_s016 + 1.78905 w_k1_MAT_s016
                                 + 1.38592 w_k1_MSFT_s016
                                 + 1.09265 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.83886 w_k1_ADM_s017
                                 + 1.03145 w_k1_AXP_s017
                                 + 0.962839 w_k1_DIS_s017
                                 + 0.984368 w_k1_FDX_s017
                                 + 0.87344 w_k1_IBM_s017
                                 + 0.948032 w_k1_INTC_s017
                                 + 0.863338 w_k1_KO_s017
                                 + 1.04507 w_k1_MAT_s017
                                 + 0.836993 w_k1_MSFT_s017
                                 + 1.02802 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.726194 w_k1_ADM_s017
                                 + 0.84378 w_k1_AXP_s017
                                 + 0.779286 w_k1_DIS_s017
                                 + 0.757756 w_k1_FDX_s017
                                 + 1.11897 w_k1_IBM_s017
                                 + 0.882358 w_k1_INTC_s017
                                 + 0.946337 w_k1_KO_s017
                                 + 0.679707 w_k1_MAT_s017
                                 + 0.742653 w_k1_MSFT_s017
                                 + 0.827827 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.810569 w_k1_ADM_s017
                                 + 0.791966 w_k1_AXP_s017
                                 + 0.7821 w_k1_DIS_s017
                                 + 0.836118 w_k1_FDX_s017
                                 + 0.707814 w_k1_IBM_s017
                                 + 0.91738 w_k1_INTC_s017
                                 + 0.77694 w_k1_KO_s017
                                 + 0.731473 w_k1_MAT_s017
                                 + 0.838183 w_k1_MSFT_s017
                                 + 0.8233 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.93961 w_k1_ADM_s017
                                 + 1.04903 w_k1_AXP_s017
                                 + 1.15436 w_k1_DIS_s017
                                 + 0.970346 w_k1_FDX_s017
                                 + 1.02115 w_k1_IBM_s017
                                 + 0.992789 w_k1_INTC_s017
                                 + 1.23914 w_k1_KO_s017 + 1.65956 w_k1_MAT_s017
                                 + 0.998154 w_k1_MSFT_s017
                                 + 1.02878 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.11833 w_k1_ADM_s017
                                 + 1.52075 w_k1_AXP_s017
                                 + 1.18932 w_k1_DIS_s017 + 1.1106 w_k1_FDX_s017
                                 + 1.68879 w_k1_IBM_s017
                                 + 1.20438 w_k1_INTC_s017
                                 + 1.21484 w_k1_KO_s017 + 1.70855 w_k1_MAT_s017
                                 + 0.971143 w_k1_MSFT_s017
                                 + 1.09159 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.954106 w_k1_ADM_s017
                                 + 1.52482 w_k1_AXP_s017
                                 + 1.44843 w_k1_DIS_s017
                                 + 1.34543 w_k1_FDX_s017
                                 + 0.801385 w_k1_IBM_s017
                                 + 1.03236 w_k1_INTC_s017
                                 + 1.15772 w_k1_KO_s017
                                 + 0.887109 w_k1_MAT_s017
                                 + 1.00634 w_k1_MSFT_s017
                                 + 1.09421 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.27244 w_k1_ADM_s017
                                 + 2.16899 w_k1_AXP_s017
                                 + 1.96338 w_k1_DIS_s017
                                 + 1.52743 w_k1_FDX_s017
                                 + 1.11163 w_k1_IBM_s017
                                 + 1.4349 w_k1_INTC_s017 + 1.50169 w_k1_KO_s017
                                 + 1.85512 w_k1_MAT_s017
                                 + 1.2979 w_k1_MSFT_s017 + 1.32442 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.09655 w_k1_ADM_s017
                                 + 1.09208 w_k1_AXP_s017
                                 + 1.67572 w_k1_DIS_s017
                                 + 1.15074 w_k1_FDX_s017
                                 + 1.48406 w_k1_IBM_s017
                                 + 1.27013 w_k1_INTC_s017
                                 + 1.49927 w_k1_KO_s017
                                 + 0.988105 w_k1_MAT_s017
                                 + 1.05654 w_k1_MSFT_s017
                                 + 1.13089 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04198 w_k1_ADM_s100
                                 + 0.904969 w_k1_AXP_s100
                                 + 1.1379 w_k1_DIS_s100 + 1.14633 w_k1_FDX_s100
                                 + 0.809523 w_k1_IBM_s100
                                 + 1.01007 w_k1_INTC_s100
                                 + 0.933213 w_k1_KO_s100
                                 + 0.94465 w_k1_MAT_s100
                                 + 1.11792 w_k1_MSFT_s100
                                 + 1.00166 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.907173 w_k1_ADM_s100
                                 + 0.935533 w_k1_AXP_s100
                                 + 0.959384 w_k1_DIS_s100
                                 + 0.977822 w_k1_FDX_s100
                                 + 1.19799 w_k1_IBM_s100
                                 + 0.873721 w_k1_INTC_s100
                                 + 0.896124 w_k1_KO_s100
                                 + 0.98121 w_k1_MAT_s100
                                 + 0.885572 w_k1_MSFT_s100
                                 + 1.01513 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.08205 w_k1_ADM_s100
                                 + 1.28133 w_k1_AXP_s100 + 1.2708 w_k1_DIS_s100
                                 + 1.02575 w_k1_FDX_s100
                                 + 1.92698 w_k1_IBM_s100
                                 + 0.981776 w_k1_INTC_s100
                                 + 1.31547 w_k1_KO_s100 + 1.88017 w_k1_MAT_s100
                                 + 1.07611 w_k1_MSFT_s100 + 1.0552 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.26146 w_k1_ADM_s100
                                 + 1.29718 w_k1_AXP_s100 + 1.6566 w_k1_DIS_s100
                                 + 1.45109 w_k1_FDX_s100
                                 + 0.952502 w_k1_IBM_s100
                                 + 1.23665 w_k1_INTC_s100
                                 + 1.16482 w_k1_KO_s100 + 1.18172 w_k1_MAT_s100
                                 + 1.36359 w_k1_MSFT_s100
                                 + 1.14962 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.955133 w_k1_ADM_s100
                                 + 0.955583 w_k1_AXP_s100
                                 + 1.04219 w_k1_DIS_s100
                                 + 0.939658 w_k1_FDX_s100
                                 + 1.44815 w_k1_IBM_s100
                                 + 0.839396 w_k1_INTC_s100
                                 + 1.07121 w_k1_KO_s100 + 1.44747 w_k1_MAT_s100
                                 + 1.01028 w_k1_MSFT_s100
                                 + 1.06286 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.40271 w_k1_ADM_s100
                                 + 1.2112 w_k1_AXP_s100 + 1.3381 w_k1_DIS_s100
                                 + 1.25198 w_k1_FDX_s100
                                 + 1.23907 w_k1_IBM_s100
                                 + 1.10932 w_k1_INTC_s100
                                 + 1.17958 w_k1_KO_s100 + 1.15708 w_k1_MAT_s100
                                 + 1.16879 w_k1_MSFT_s100 + 1.1929 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.766639 w_k1_ADM_s100
                                 + 0.742046 w_k1_AXP_s100
                                 + 0.726761 w_k1_DIS_s100
                                 + 0.815857 w_k1_FDX_s100
                                 + 0.872114 w_k1_IBM_s100
                                 + 0.79169 w_k1_INTC_s100
                                 + 0.752344 w_k1_KO_s100
                                 + 0.655101 w_k1_MAT_s100
                                 + 0.873766 w_k1_MSFT_s100
                                 + 0.852096 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.04469 w_k1_ADM_s100
                                 + 0.792635 w_k1_AXP_s100
                                 + 0.912866 w_k1_DIS_s100
                                 + 1.05089 w_k1_FDX_s100
                                 + 0.614178 w_k1_IBM_s100
                                 + 0.809046 w_k1_INTC_s100
                                 + 0.716364 w_k1_KO_s100
                                 + 0.68647 w_k1_MAT_s100
                                 + 0.98806 w_k1_MSFT_s100
                                 + 0.893262 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.58896 w_k1_ADM_s101
                                 + 0.719525 w_k1_AXP_s101
                                 + 0.730049 w_k1_DIS_s101
                                 + 0.743198 w_k1_FDX_s101
                                 + 0.996157 w_k1_IBM_s101
                                 + 0.90554 w_k1_INTC_s101
                                 + 0.842222 w_k1_KO_s101
                                 + 0.770173 w_k1_MAT_s101
                                 + 0.891495 w_k1_MSFT_s101
                                 + 0.825226 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.669714 w_k1_ADM_s101
                                 + 0.621365 w_k1_AXP_s101
                                 + 0.643316 w_k1_DIS_s101
                                 + 0.73305 w_k1_FDX_s101
                                 + 0.693596 w_k1_IBM_s101
                                 + 0.888883 w_k1_INTC_s101
                                 + 0.713651 w_k1_KO_s101
                                 + 0.438772 w_k1_MAT_s101
                                 + 0.865552 w_k1_MSFT_s101
                                 + 0.688408 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.07253 w_k1_ADM_s101
                                 + 0.992879 w_k1_AXP_s101
                                 + 1.17288 w_k1_DIS_s101 + 1.104 w_k1_FDX_s101
                                 + 0.835949 w_k1_IBM_s101
                                 + 1.04209 w_k1_INTC_s101
                                 + 0.942145 w_k1_KO_s101
                                 + 0.792259 w_k1_MAT_s101
                                 + 1.20062 w_k1_MSFT_s101
                                 + 0.941064 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.737216 w_k1_ADM_s101
                                 + 1.14183 w_k1_AXP_s101
                                 + 0.967443 w_k1_DIS_s101
                                 + 1.03478 w_k1_FDX_s101
                                 + 0.781114 w_k1_IBM_s101
                                 + 0.963033 w_k1_INTC_s101
                                 + 0.979665 w_k1_KO_s101
                                 + 0.989868 w_k1_MAT_s101
                                 + 1.09239 w_k1_MSFT_s101
                                 + 0.863078 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.937415 w_k1_ADM_s101
                                 + 1.06772 w_k1_AXP_s101
                                 + 1.22452 w_k1_DIS_s101
                                 + 1.00196 w_k1_FDX_s101
                                 + 1.23982 w_k1_IBM_s101
                                 + 0.968147 w_k1_INTC_s101
                                 + 1.08394 w_k1_KO_s101 + 1.15035 w_k1_MAT_s101
                                 + 1.04218 w_k1_MSFT_s101 + 0.9856 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.910159 w_k1_ADM_s101
                                 + 0.750001 w_k1_AXP_s101
                                 + 1.0094 w_k1_DIS_s101
                                 + 0.899417 w_k1_FDX_s101
                                 + 0.883966 w_k1_IBM_s101
                                 + 1.00998 w_k1_INTC_s101
                                 + 1.07155 w_k1_KO_s101
                                 + 0.822203 w_k1_MAT_s101
                                 + 1.02112 w_k1_MSFT_s101
                                 + 0.952658 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.23702 w_k1_ADM_s101
                                 + 1.17713 w_k1_AXP_s101
                                 + 1.41509 w_k1_DIS_s101
                                 + 1.18732 w_k1_FDX_s101
                                 + 1.60381 w_k1_IBM_s101
                                 + 1.0201 w_k1_INTC_s101 + 1.49008 w_k1_KO_s101
                                 + 1.85163 w_k1_MAT_s101
                                 + 1.25571 w_k1_MSFT_s101
                                 + 1.10594 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.3179 w_k1_ADM_s101
                                 + 1.73667 w_k1_AXP_s101 + 1.9898 w_k1_DIS_s101
                                 + 1.46766 w_k1_FDX_s101
                                 + 1.16082 w_k1_IBM_s101
                                 + 1.22801 w_k1_INTC_s101
                                 + 1.26517 w_k1_KO_s101 + 1.45652 w_k1_MAT_s101
                                 + 1.34217 w_k1_MSFT_s101
                                 + 1.16648 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.738299 w_k1_ADM_s102
                                 + 0.741186 w_k1_AXP_s102
                                 + 0.646416 w_k1_DIS_s102
                                 + 0.794176 w_k1_FDX_s102
                                 + 0.691251 w_k1_IBM_s102
                                 + 0.88583 w_k1_INTC_s102
                                 + 0.687652 w_k1_KO_s102
                                 + 0.769377 w_k1_MAT_s102
                                 + 0.724054 w_k1_MSFT_s102
                                 + 0.794265 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.762564 w_k1_ADM_s102
                                 + 0.983679 w_k1_AXP_s102
                                 + 0.773001 w_k1_DIS_s102
                                 + 0.845614 w_k1_FDX_s102
                                 + 1.02025 w_k1_IBM_s102
                                 + 0.93053 w_k1_INTC_s102
                                 + 0.931723 w_k1_KO_s102
                                 + 0.979267 w_k1_MAT_s102
                                 + 0.751075 w_k1_MSFT_s102
                                 + 0.983941 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.817068 w_k1_ADM_s102
                                 + 1.13605 w_k1_AXP_s102
                                 + 1.04299 w_k1_DIS_s102
                                 + 0.941352 w_k1_FDX_s102
                                 + 1.53329 w_k1_IBM_s102
                                 + 0.945757 w_k1_INTC_s102
                                 + 1.05937 w_k1_KO_s102 + 1.38806 w_k1_MAT_s102
                                 + 0.863877 w_k1_MSFT_s102
                                 + 1.03864 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.725727 w_k1_ADM_s102
                                 + 0.669395 w_k1_AXP_s102
                                 + 0.496876 w_k1_DIS_s102
                                 + 0.621351 w_k1_FDX_s102
                                 + 0.908592 w_k1_IBM_s102
                                 + 0.702042 w_k1_INTC_s102
                                 + 0.76435 w_k1_KO_s102
                                 + 0.663518 w_k1_MAT_s102
                                 + 0.604153 w_k1_MSFT_s102
                                 + 0.72948 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.832597 w_k1_ADM_s102
                                 + 0.857431 w_k1_AXP_s102
                                 + 0.866312 w_k1_DIS_s102
                                 + 0.969645 w_k1_FDX_s102
                                 + 0.857895 w_k1_IBM_s102
                                 + 1.07337 w_k1_INTC_s102
                                 + 0.899656 w_k1_KO_s102
                                 + 0.774643 w_k1_MAT_s102
                                 + 0.858219 w_k1_MSFT_s102
                                 + 0.984544 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.06888 w_k1_ADM_s102
                                 + 1.29151 w_k1_AXP_s102
                                 + 1.24228 w_k1_DIS_s102
                                 + 1.15056 w_k1_FDX_s102
                                 + 1.20969 w_k1_IBM_s102
                                 + 1.18889 w_k1_INTC_s102
                                 + 1.20223 w_k1_KO_s102 + 1.16166 w_k1_MAT_s102
                                 + 1.14621 w_k1_MSFT_s102
                                 + 1.04682 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.09958 w_k1_ADM_s102
                                 + 1.7212 w_k1_AXP_s102 + 1.30216 w_k1_DIS_s102
                                 + 1.29224 w_k1_FDX_s102
                                 + 0.986244 w_k1_IBM_s102
                                 + 1.02453 w_k1_INTC_s102
                                 + 1.10164 w_k1_KO_s102 + 1.46049 w_k1_MAT_s102
                                 + 0.944269 w_k1_MSFT_s102
                                 + 1.1406 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.781242 w_k1_ADM_s102
                                 + 0.854617 w_k1_AXP_s102
                                 + 0.775953 w_k1_DIS_s102
                                 + 0.785665 w_k1_FDX_s102
                                 + 1.1878 w_k1_IBM_s102
                                 + 0.935747 w_k1_INTC_s102
                                 + 0.957014 w_k1_KO_s102
                                 + 1.38402 w_k1_MAT_s102
                                 + 0.906942 w_k1_MSFT_s102
                                 + 0.901415 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.847794 w_k1_ADM_s103
                                 + 0.636822 w_k1_AXP_s103
                                 + 0.815158 w_k1_DIS_s103
                                 + 0.913157 w_k1_FDX_s103
                                 + 0.61415 w_k1_IBM_s103
                                 + 0.982581 w_k1_INTC_s103
                                 + 0.814198 w_k1_KO_s103
                                 + 0.452503 w_k1_MAT_s103
                                 + 0.941809 w_k1_MSFT_s103
                                 + 0.824039 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.828021 w_k1_ADM_s103
                                 + 0.849544 w_k1_AXP_s103
                                 + 0.848858 w_k1_DIS_s103
                                 + 0.792526 w_k1_FDX_s103
                                 + 1.17265 w_k1_IBM_s103
                                 + 0.932639 w_k1_INTC_s103
                                 + 1.02704 w_k1_KO_s103 + 1.20969 w_k1_MAT_s103
                                 + 0.902848 w_k1_MSFT_s103
                                 + 0.855728 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.876862 w_k1_ADM_s103
                                 + 1.57842 w_k1_AXP_s103
                                 + 1.47433 w_k1_DIS_s103
                                 + 1.31934 w_k1_FDX_s103
                                 + 1.47128 w_k1_IBM_s103
                                 + 1.41703 w_k1_INTC_s103
                                 + 1.37523 w_k1_KO_s103 + 1.86883 w_k1_MAT_s103
                                 + 1.19305 w_k1_MSFT_s103
                                 + 1.05081 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.811262 w_k1_ADM_s103
                                 + 1.16974 w_k1_AXP_s103
                                 + 1.17586 w_k1_DIS_s103
                                 + 1.01145 w_k1_FDX_s103
                                 + 1.14266 w_k1_IBM_s103
                                 + 1.13554 w_k1_INTC_s103
                                 + 1.26595 w_k1_KO_s103 + 1.41384 w_k1_MAT_s103
                                 + 1.21088 w_k1_MSFT_s103
                                 + 0.99852 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.812865 w_k1_ADM_s103
                                 + 0.751177 w_k1_AXP_s103
                                 + 0.740371 w_k1_DIS_s103
                                 + 0.814331 w_k1_FDX_s103
                                 + 0.751366 w_k1_IBM_s103
                                 + 0.898959 w_k1_INTC_s103
                                 + 0.828051 w_k1_KO_s103
                                 + 0.822664 w_k1_MAT_s103
                                 + 0.913414 w_k1_MSFT_s103
                                 + 0.865967 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.14345 w_k1_ADM_s103
                                 + 1.17194 w_k1_AXP_s103
                                 + 1.30929 w_k1_DIS_s103
                                 + 1.23054 w_k1_FDX_s103
                                 + 0.92981 w_k1_IBM_s103
                                 + 1.17097 w_k1_INTC_s103
                                 + 1.06487 w_k1_KO_s103
                                 + 0.755129 w_k1_MAT_s103
                                 + 1.13026 w_k1_MSFT_s103
                                 + 1.04712 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.08722 w_k1_ADM_s103
                                 + 1.20934 w_k1_AXP_s103
                                 + 0.973031 w_k1_DIS_s103
                                 + 0.97924 w_k1_FDX_s103
                                 + 1.65125 w_k1_IBM_s103
                                 + 1.10918 w_k1_INTC_s103
                                 + 1.34317 w_k1_KO_s103 + 1.62206 w_k1_MAT_s103
                                 + 1.04105 w_k1_MSFT_s103
                                 + 1.04363 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.933276 w_k1_ADM_s103
                                 + 1.0252 w_k1_AXP_s103
                                 + 0.976354 w_k1_DIS_s103
                                 + 1.08124 w_k1_FDX_s103
                                 + 1.07123 w_k1_IBM_s103
                                 + 1.05792 w_k1_INTC_s103
                                 + 0.943858 w_k1_KO_s103
                                 + 1.13679 w_k1_MAT_s103
                                 + 1.05459 w_k1_MSFT_s103
                                 + 1.01311 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.974885 w_k1_ADM_s104
                                 + 1.1201 w_k1_AXP_s104 + 1.14125 w_k1_DIS_s104
                                 + 0.961159 w_k1_FDX_s104
                                 + 1.64798 w_k1_IBM_s104
                                 + 1.1989 w_k1_INTC_s104 + 1.21386 w_k1_KO_s104
                                 + 1.19963 w_k1_MAT_s104
                                 + 0.93138 w_k1_MSFT_s104
                                 + 1.12669 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.978651 w_k1_ADM_s104
                                 + 1.07539 w_k1_AXP_s104 + 1.026 w_k1_DIS_s104
                                 + 1.02515 w_k1_FDX_s104
                                 + 1.10333 w_k1_IBM_s104
                                 + 1.01133 w_k1_INTC_s104
                                 + 0.967099 w_k1_KO_s104
                                 + 1.00765 w_k1_MAT_s104
                                 + 0.940746 w_k1_MSFT_s104
                                 + 1.02349 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.884963 w_k1_ADM_s104
                                 + 0.784453 w_k1_AXP_s104
                                 + 0.748613 w_k1_DIS_s104
                                 + 0.857511 w_k1_FDX_s104
                                 + 1.00295 w_k1_IBM_s104
                                 + 0.959578 w_k1_INTC_s104
                                 + 0.928197 w_k1_KO_s104
                                 + 1.02142 w_k1_MAT_s104
                                 + 0.931038 w_k1_MSFT_s104
                                 + 0.918973 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.09549 w_k1_ADM_s104
                                 + 1.00482 w_k1_AXP_s104
                                 + 1.00503 w_k1_DIS_s104
                                 + 1.05267 w_k1_FDX_s104
                                 + 0.801628 w_k1_IBM_s104
                                 + 1.05183 w_k1_INTC_s104
                                 + 0.865764 w_k1_KO_s104
                                 + 0.69582 w_k1_MAT_s104
                                 + 0.853322 w_k1_MSFT_s104
                                 + 1.00891 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.27158 w_k1_ADM_s104
                                 + 1.00028 w_k1_AXP_s104
                                 + 1.36241 w_k1_DIS_s104
                                 + 1.20649 w_k1_FDX_s104
                                 + 0.877171 w_k1_IBM_s104
                                 + 0.969819 w_k1_INTC_s104
                                 + 1.05242 w_k1_KO_s104
                                 + 0.970993 w_k1_MAT_s104
                                 + 1.07214 w_k1_MSFT_s104
                                 + 1.08868 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.778003 w_k1_ADM_s104
                                 + 0.627181 w_k1_AXP_s104
                                 + 0.656418 w_k1_DIS_s104
                                 + 0.920949 w_k1_FDX_s104
                                 + 0.581204 w_k1_IBM_s104
                                 + 0.984237 w_k1_INTC_s104
                                 + 0.651447 w_k1_KO_s104
                                 + 0.421704 w_k1_MAT_s104
                                 + 0.852679 w_k1_MSFT_s104
                                 + 0.792691 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.26564 w_k1_ADM_s104
                                 + 1.45371 w_k1_AXP_s104
                                 + 1.46541 w_k1_DIS_s104
                                 + 1.38127 w_k1_FDX_s104 + 1.1246 w_k1_IBM_s104
                                 + 1.1888 w_k1_INTC_s104 + 1.07536 w_k1_KO_s104
                                 + 0.962962 w_k1_MAT_s104
                                 + 1.10429 w_k1_MSFT_s104
                                 + 1.13301 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.712717 w_k1_ADM_s104
                                 + 0.761729 w_k1_AXP_s104
                                 + 0.610651 w_k1_DIS_s104
                                 + 0.696366 w_k1_FDX_s104
                                 + 1.03542 w_k1_IBM_s104
                                 + 0.796035 w_k1_INTC_s104
                                 + 0.769859 w_k1_KO_s104
                                 + 1.06631 w_k1_MAT_s104
                                 + 0.698758 w_k1_MSFT_s104
                                 + 0.844212 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.855429 w_k1_ADM_s105
                                 + 0.722349 w_k1_AXP_s105
                                 + 0.566268 w_k1_DIS_s105
                                 + 0.767523 w_k1_FDX_s105
                                 + 0.755878 w_k1_IBM_s105
                                 + 0.92571 w_k1_INTC_s105
                                 + 0.802416 w_k1_KO_s105
                                 + 0.623098 w_k1_MAT_s105
                                 + 0.849064 w_k1_MSFT_s105
                                 + 0.809598 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.24405 w_k1_ADM_s105
                                 + 1.08655 w_k1_AXP_s105
                                 + 1.22507 w_k1_DIS_s105 + 1.2275 w_k1_FDX_s105
                                 + 0.813089 w_k1_IBM_s105
                                 + 1.37726 w_k1_INTC_s105
                                 + 1.18855 w_k1_KO_s105 + 0.69506 w_k1_MAT_s105
                                 + 1.26971 w_k1_MSFT_s105
                                 + 1.12175 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.4101 w_k1_ADM_s105
                                 + 1.54197 w_k1_AXP_s105 + 1.587 w_k1_DIS_s105
                                 + 1.39262 w_k1_FDX_s105
                                 + 1.37973 w_k1_IBM_s105
                                 + 1.75951 w_k1_INTC_s105
                                 + 1.48977 w_k1_KO_s105 + 1.18868 w_k1_MAT_s105
                                 + 1.46064 w_k1_MSFT_s105
                                 + 1.21631 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.781793 w_k1_ADM_s105
                                 + 1.09134 w_k1_AXP_s105
                                 + 0.826151 w_k1_DIS_s105
                                 + 0.88502 w_k1_FDX_s105
                                 + 1.31572 w_k1_IBM_s105
                                 + 1.09833 w_k1_INTC_s105
                                 + 1.03387 w_k1_KO_s105 + 1.22355 w_k1_MAT_s105
                                 + 0.909247 w_k1_MSFT_s105
                                 + 1.08594 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.16099 w_k1_ADM_s105
                                 + 0.991767 w_k1_AXP_s105
                                 + 1.07468 w_k1_DIS_s105
                                 + 1.02286 w_k1_FDX_s105
                                 + 1.29937 w_k1_IBM_s105
                                 + 1.31446 w_k1_INTC_s105 + 1.2277 w_k1_KO_s105
                                 + 1.01021 w_k1_MAT_s105 + 1.202 w_k1_MSFT_s105
                                 + 1.17824 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.984266 w_k1_ADM_s105
                                 + 1.33648 w_k1_AXP_s105
                                 + 1.17456 w_k1_DIS_s105
                                 + 1.07825 w_k1_FDX_s105
                                 + 1.69672 w_k1_IBM_s105
                                 + 1.29629 w_k1_INTC_s105
                                 + 1.46576 w_k1_KO_s105 + 1.73487 w_k1_MAT_s105
                                 + 1.0877 w_k1_MSFT_s105 + 1.27968 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.823362 w_k1_ADM_s105
                                 + 0.812713 w_k1_AXP_s105
                                 + 0.804636 w_k1_DIS_s105
                                 + 0.859965 w_k1_FDX_s105
                                 + 1.00764 w_k1_IBM_s105
                                 + 1.09272 w_k1_INTC_s105
                                 + 0.933383 w_k1_KO_s105
                                 + 0.754501 w_k1_MAT_s105
                                 + 0.986494 w_k1_MSFT_s105
                                 + 0.953384 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.1192 w_k1_ADM_s105
                                 + 1.22193 w_k1_AXP_s105
                                 + 0.945069 w_k1_DIS_s105
                                 + 1.13098 w_k1_FDX_s105
                                 + 0.90384 w_k1_IBM_s105
                                 + 1.03915 w_k1_INTC_s105
                                 + 0.947651 w_k1_KO_s105
                                 + 1.11182 w_k1_MAT_s105
                                 + 1.02592 w_k1_MSFT_s105
                                 + 1.09045 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.760446 w_k1_ADM_s106
                                 + 0.911322 w_k1_AXP_s106
                                 + 1.30217 w_k1_DIS_s106
                                 + 1.11352 w_k1_FDX_s106
                                 + 1.03994 w_k1_IBM_s106
                                 + 1.21492 w_k1_INTC_s106 + 1.1852 w_k1_KO_s106
                                 + 1.12823 w_k1_MAT_s106
                                 + 1.23582 w_k1_MSFT_s106
                                 + 1.07589 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.23118 w_k1_ADM_s106
                                 + 1.38504 w_k1_AXP_s106
                                 + 1.56495 w_k1_DIS_s106
                                 + 1.38263 w_k1_FDX_s106
                                 + 1.24322 w_k1_IBM_s106
                                 + 1.22211 w_k1_INTC_s106
                                 + 1.27353 w_k1_KO_s106 + 1.27297 w_k1_MAT_s106
                                 + 1.38439 w_k1_MSFT_s106
                                 + 1.15363 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.05919 w_k1_ADM_s106
                                 + 0.999015 w_k1_AXP_s106
                                 + 0.887994 w_k1_DIS_s106
                                 + 0.899824 w_k1_FDX_s106
                                 + 1.4419 w_k1_IBM_s106
                                 + 0.964102 w_k1_INTC_s106
                                 + 0.999079 w_k1_KO_s106
                                 + 1.27787 w_k1_MAT_s106
                                 + 0.948551 w_k1_MSFT_s106
                                 + 1.11551 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.709012 w_k1_ADM_s106
                                 + 0.666496 w_k1_AXP_s106
                                 + 0.80597 w_k1_DIS_s106
                                 + 0.88211 w_k1_FDX_s106
                                 + 0.71759 w_k1_IBM_s106
                                 + 0.833557 w_k1_INTC_s106
                                 + 0.733817 w_k1_KO_s106
                                 + 0.808919 w_k1_MAT_s106
                                 + 0.915825 w_k1_MSFT_s106
                                 + 0.912742 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.21319 w_k1_ADM_s106
                                 + 0.791951 w_k1_AXP_s106
                                 + 1.07073 w_k1_DIS_s106
                                 + 1.04044 w_k1_FDX_s106
                                 + 0.828198 w_k1_IBM_s106
                                 + 1.10262 w_k1_INTC_s106
                                 + 0.82085 w_k1_KO_s106
                                 + 0.649607 w_k1_MAT_s106
                                 + 1.15727 w_k1_MSFT_s106
                                 + 1.03774 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.08192 w_k1_ADM_s106
                                 + 1.04863 w_k1_AXP_s106
                                 + 0.777847 w_k1_DIS_s106
                                 + 0.904832 w_k1_FDX_s106
                                 + 0.924847 w_k1_IBM_s106
                                 + 0.817489 w_k1_INTC_s106
                                 + 0.841535 w_k1_KO_s106
                                 + 1.31662 w_k1_MAT_s106
                                 + 0.887962 w_k1_MSFT_s106
                                 + 0.978519 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.19895 w_k1_ADM_s106
                                 + 1.32265 w_k1_AXP_s106 + 1.1324 w_k1_DIS_s106
                                 + 1.15288 w_k1_FDX_s106
                                 + 1.07223 w_k1_IBM_s106
                                 + 0.902742 w_k1_INTC_s106
                                 + 0.971857 w_k1_KO_s106
                                 + 1.13649 w_k1_MAT_s106
                                 + 1.09321 w_k1_MSFT_s106
                                 + 1.17169 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.10718 w_k1_ADM_s106
                                 + 1.49581 w_k1_AXP_s106
                                 + 1.41117 w_k1_DIS_s106
                                 + 1.13831 w_k1_FDX_s106
                                 + 1.87905 w_k1_IBM_s106
                                 + 1.14048 w_k1_INTC_s106
                                 + 1.45251 w_k1_KO_s106 + 2.33881 w_k1_MAT_s106
                                 + 1.27459 w_k1_MSFT_s106
                                 + 1.26392 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.836509 w_k1_ADM_s107
                                 + 0.700585 w_k1_AXP_s107
                                 + 0.994041 w_k1_DIS_s107
                                 + 0.970031 w_k1_FDX_s107
                                 + 0.690608 w_k1_IBM_s107
                                 + 0.822012 w_k1_INTC_s107
                                 + 0.868026 w_k1_KO_s107
                                 + 0.52076 w_k1_MAT_s107
                                 + 1.0465 w_k1_MSFT_s107 + 0.82454 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.09839 w_k1_ADM_s107
                                 + 1.00296 w_k1_AXP_s107
                                 + 1.09018 w_k1_DIS_s107
                                 + 1.24123 w_k1_FDX_s107
                                 + 0.694541 w_k1_IBM_s107
                                 + 1.18697 w_k1_INTC_s107
                                 + 0.754204 w_k1_KO_s107
                                 + 0.435101 w_k1_MAT_s107
                                 + 1.04984 w_k1_MSFT_s107
                                 + 0.927503 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.686499 w_k1_ADM_s107
                                 + 1.05983 w_k1_AXP_s107
                                 + 0.931146 w_k1_DIS_s107
                                 + 0.912727 w_k1_FDX_s107
                                 + 1.1479 w_k1_IBM_s107
                                 + 0.939998 w_k1_INTC_s107
                                 + 0.823914 w_k1_KO_s107
                                 + 1.00455 w_k1_MAT_s107
                                 + 1.01758 w_k1_MSFT_s107
                                 + 0.917207 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.822831 w_k1_ADM_s107
                                 + 0.892911 w_k1_AXP_s107
                                 + 0.802843 w_k1_DIS_s107
                                 + 0.947439 w_k1_FDX_s107
                                 + 0.654867 w_k1_IBM_s107
                                 + 1.09964 w_k1_INTC_s107
                                 + 0.781592 w_k1_KO_s107
                                 + 0.682472 w_k1_MAT_s107
                                 + 0.882775 w_k1_MSFT_s107
                                 + 0.92417 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.920734 w_k1_ADM_s107
                                 + 1.03645 w_k1_AXP_s107
                                 + 1.24301 w_k1_DIS_s107
                                 + 1.02107 w_k1_FDX_s107
                                 + 0.916515 w_k1_IBM_s107
                                 + 1.02278 w_k1_INTC_s107
                                 + 0.919504 w_k1_KO_s107
                                 + 0.825406 w_k1_MAT_s107
                                 + 1.14229 w_k1_MSFT_s107
                                 + 0.990879 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.871506 w_k1_ADM_s107
                                 + 0.755706 w_k1_AXP_s107
                                 + 0.74082 w_k1_DIS_s107
                                 + 0.729701 w_k1_FDX_s107
                                 + 0.880477 w_k1_IBM_s107
                                 + 0.636548 w_k1_INTC_s107
                                 + 0.76133 w_k1_KO_s107
                                 + 0.926768 w_k1_MAT_s107
                                 + 0.809105 w_k1_MSFT_s107
                                 + 0.850715 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.21562 w_k1_ADM_s107
                                 + 1.52231 w_k1_AXP_s107
                                 + 1.47396 w_k1_DIS_s107
                                 + 1.19992 w_k1_FDX_s107
                                 + 1.46006 w_k1_IBM_s107
                                 + 0.909136 w_k1_INTC_s107
                                 + 1.20639 w_k1_KO_s107 + 2.13758 w_k1_MAT_s107
                                 + 1.13355 w_k1_MSFT_s107
                                 + 1.25539 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.637623 w_k1_ADM_s107
                                 + 0.699081 w_k1_AXP_s107
                                 + 0.609319 w_k1_DIS_s107
                                 + 0.774765 w_k1_FDX_s107
                                 + 0.463368 w_k1_IBM_s107
                                 + 0.718362 w_k1_INTC_s107
                                 + 0.544818 w_k1_KO_s107
                                 + 0.378066 w_k1_MAT_s107
                                 + 0.762021 w_k1_MSFT_s107
                                 + 0.815711 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.00105 w_k1_ADM_s110
                                 + 1.14134 w_k1_AXP_s110
                                 + 0.964023 w_k1_DIS_s110
                                 + 1.0188 w_k1_FDX_s110 + 1.40309 w_k1_IBM_s110
                                 + 0.974659 w_k1_INTC_s110
                                 + 1.1441 w_k1_KO_s110 + 2.02188 w_k1_MAT_s110
                                 + 0.945127 w_k1_MSFT_s110
                                 + 1.25726 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01141 w_k1_ADM_s110
                                 + 1.37759 w_k1_AXP_s110
                                 + 1.39126 w_k1_DIS_s110
                                 + 1.20842 w_k1_FDX_s110 + 1.063 w_k1_IBM_s110
                                 + 1.07345 w_k1_INTC_s110
                                 + 1.06161 w_k1_KO_s110 + 1.28085 w_k1_MAT_s110
                                 + 1.11414 w_k1_MSFT_s110 + 1.2356 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.21813 w_k1_ADM_s110
                                 + 1.35226 w_k1_AXP_s110 + 1.5651 w_k1_DIS_s110
                                 + 1.23704 w_k1_FDX_s110
                                 + 1.45079 w_k1_IBM_s110
                                 + 1.0735 w_k1_INTC_s110 + 1.29301 w_k1_KO_s110
                                 + 1.75501 w_k1_MAT_s110
                                 + 1.37459 w_k1_MSFT_s110 + 1.3192 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.14689 w_k1_ADM_s110
                                 + 0.969755 w_k1_AXP_s110
                                 + 0.804669 w_k1_DIS_s110
                                 + 0.864317 w_k1_FDX_s110
                                 + 1.18895 w_k1_IBM_s110
                                 + 0.873061 w_k1_INTC_s110
                                 + 0.981755 w_k1_KO_s110
                                 + 1.33167 w_k1_MAT_s110
                                 + 0.833387 w_k1_MSFT_s110
                                 + 1.06812 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.15765 w_k1_ADM_s110
                                 + 1.06803 w_k1_AXP_s110
                                 + 1.48278 w_k1_DIS_s110
                                 + 1.20039 w_k1_FDX_s110
                                 + 0.792462 w_k1_IBM_s110
                                 + 1.16161 w_k1_INTC_s110
                                 + 1.10926 w_k1_KO_s110
                                 + 0.785024 w_k1_MAT_s110
                                 + 1.22018 w_k1_MSFT_s110
                                 + 1.16512 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.9637 w_k1_ADM_s110
                                 + 1.033 w_k1_AXP_s110 + 0.941771 w_k1_DIS_s110
                                 + 0.978303 w_k1_FDX_s110
                                 + 0.888211 w_k1_IBM_s110
                                 + 0.982031 w_k1_INTC_s110
                                 + 0.972593 w_k1_KO_s110
                                 + 1.10023 w_k1_MAT_s110
                                 + 1.01626 w_k1_MSFT_s110
                                 + 1.00324 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.873273 w_k1_ADM_s110
                                 + 0.774528 w_k1_AXP_s110
                                 + 0.795618 w_k1_DIS_s110
                                 + 0.94854 w_k1_FDX_s110
                                 + 0.596322 w_k1_IBM_s110
                                 + 0.892342 w_k1_INTC_s110
                                 + 0.747008 w_k1_KO_s110
                                 + 0.711915 w_k1_MAT_s110
                                 + 0.915087 w_k1_MSFT_s110
                                 + 1.01231 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.779674 w_k1_ADM_s110
                                 + 0.795111 w_k1_AXP_s110
                                 + 0.612662 w_k1_DIS_s110
                                 + 0.765254 w_k1_FDX_s110
                                 + 0.819839 w_k1_IBM_s110
                                 + 0.863002 w_k1_INTC_s110
                                 + 0.828475 w_k1_KO_s110
                                 + 0.920217 w_k1_MAT_s110
                                 + 0.79873 w_k1_MSFT_s110
                                 + 0.871371 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04559 w_k1_ADM_s111
                                 + 0.972773 w_k1_AXP_s111
                                 + 0.889783 w_k1_DIS_s111
                                 + 0.905726 w_k1_FDX_s111
                                 + 1.09103 w_k1_IBM_s111
                                 + 0.872745 w_k1_INTC_s111
                                 + 0.835033 w_k1_KO_s111
                                 + 1.08599 w_k1_MAT_s111
                                 + 0.897154 w_k1_MSFT_s111
                                 + 1.10833 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.2172 w_k1_ADM_s111
                                 + 1.43213 w_k1_AXP_s111
                                 + 1.62287 w_k1_DIS_s111
                                 + 1.28426 w_k1_FDX_s111
                                 + 1.67846 w_k1_IBM_s111
                                 + 1.03241 w_k1_INTC_s111
                                 + 1.28667 w_k1_KO_s111 + 2.12174 w_k1_MAT_s111
                                 + 1.25975 w_k1_MSFT_s111
                                 + 1.32223 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.07769 w_k1_ADM_s111
                                 + 0.901225 w_k1_AXP_s111
                                 + 1.22085 w_k1_DIS_s111
                                 + 1.03583 w_k1_FDX_s111
                                 + 0.864133 w_k1_IBM_s111
                                 + 1.11705 w_k1_INTC_s111
                                 + 0.801176 w_k1_KO_s111
                                 + 0.804503 w_k1_MAT_s111
                                 + 1.10933 w_k1_MSFT_s111
                                 + 1.03476 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.00607 w_k1_ADM_s111
                                 + 0.947674 w_k1_AXP_s111
                                 + 1.07778 w_k1_DIS_s111
                                 + 0.94421 w_k1_FDX_s111
                                 + 1.58389 w_k1_IBM_s111
                                 + 0.804801 w_k1_INTC_s111
                                 + 1.14609 w_k1_KO_s111 + 1.45134 w_k1_MAT_s111
                                 + 1.04003 w_k1_MSFT_s111
                                 + 1.08093 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03386 w_k1_ADM_s111
                                 + 0.808104 w_k1_AXP_s111
                                 + 0.943707 w_k1_DIS_s111
                                 + 1.08128 w_k1_FDX_s111
                                 + 0.657303 w_k1_IBM_s111
                                 + 0.79624 w_k1_INTC_s111
                                 + 0.755384 w_k1_KO_s111
                                 + 0.849896 w_k1_MAT_s111
                                 + 1.01961 w_k1_MSFT_s111
                                 + 0.979331 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.672373 w_k1_ADM_s111
                                 + 0.672658 w_k1_AXP_s111
                                 + 0.714321 w_k1_DIS_s111
                                 + 0.736771 w_k1_FDX_s111
                                 + 0.877142 w_k1_IBM_s111
                                 + 0.722622 w_k1_INTC_s111
                                 + 0.704316 w_k1_KO_s111
                                 + 0.92539 w_k1_MAT_s111
                                 + 0.900752 w_k1_MSFT_s111
                                 + 0.920071 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.04115 w_k1_ADM_s111
                                 + 1.34543 w_k1_AXP_s111
                                 + 1.19023 w_k1_DIS_s111
                                 + 1.18696 w_k1_FDX_s111
                                 + 1.10196 w_k1_IBM_s111
                                 + 1.08938 w_k1_INTC_s111
                                 + 0.952073 w_k1_KO_s111
                                 + 1.14232 w_k1_MAT_s111
                                 + 1.04815 w_k1_MSFT_s111
                                 + 1.17591 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.44188 w_k1_ADM_s111
                                 + 1.28778 w_k1_AXP_s111 + 1.6866 w_k1_DIS_s111
                                 + 1.35419 w_k1_FDX_s111
                                 + 1.03363 w_k1_IBM_s111
                                 + 1.06416 w_k1_INTC_s111 + 1.171 w_k1_KO_s111
                                 + 1.67589 w_k1_MAT_s111
                                 + 1.38824 w_k1_MSFT_s111
                                 + 1.32763 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.53753 w_k1_ADM_s112
                                 + 1.8056 w_k1_AXP_s112 + 1.47077 w_k1_DIS_s112
                                 + 1.44574 w_k1_FDX_s112 + 1.2527 w_k1_IBM_s112
                                 + 1.29783 w_k1_INTC_s112 + 1.2508 w_k1_KO_s112
                                 + 1.3507 w_k1_MAT_s112
                                 + 1.30929 w_k1_MSFT_s112
                                 + 1.20854 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.885857 w_k1_ADM_s112
                                 + 1.0762 w_k1_AXP_s112 + 1.03636 w_k1_DIS_s112
                                 + 0.923581 w_k1_FDX_s112
                                 + 1.49065 w_k1_IBM_s112
                                 + 1.10377 w_k1_INTC_s112
                                 + 1.14417 w_k1_KO_s112 + 1.44064 w_k1_MAT_s112
                                 + 0.956034 w_k1_MSFT_s112
                                 + 0.996473 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.17684 w_k1_ADM_s112
                                 + 1.27589 w_k1_AXP_s112
                                 + 1.29636 w_k1_DIS_s112
                                 + 1.35227 w_k1_FDX_s112
                                 + 1.04734 w_k1_IBM_s112
                                 + 1.25073 w_k1_INTC_s112
                                 + 1.16185 w_k1_KO_s112
                                 + 0.879254 w_k1_MAT_s112
                                 + 1.1991 w_k1_MSFT_s112 + 1.1614 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.11051 w_k1_ADM_s112
                                 + 0.946273 w_k1_AXP_s112
                                 + 1.14063 w_k1_DIS_s112
                                 + 1.20935 w_k1_FDX_s112
                                 + 0.679332 w_k1_IBM_s112
                                 + 1.19056 w_k1_INTC_s112
                                 + 0.973036 w_k1_KO_s112
                                 + 0.445901 w_k1_MAT_s112
                                 + 1.11765 w_k1_MSFT_s112
                                 + 1.06046 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.70346 w_k1_ADM_s112
                                 + 0.726653 w_k1_AXP_s112
                                 + 0.67999 w_k1_DIS_s112
                                 + 0.848914 w_k1_FDX_s112
                                 + 0.706192 w_k1_IBM_s112
                                 + 1.05284 w_k1_INTC_s112
                                 + 0.758576 w_k1_KO_s112
                                 + 0.609481 w_k1_MAT_s112
                                 + 0.892691 w_k1_MSFT_s112
                                 + 0.814249 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.697723 w_k1_ADM_s112
                                 + 0.901767 w_k1_AXP_s112
                                 + 0.805547 w_k1_DIS_s112
                                 + 0.810129 w_k1_FDX_s112
                                 + 1.14335 w_k1_IBM_s112
                                 + 0.903851 w_k1_INTC_s112
                                 + 1.02683 w_k1_KO_s112
                                 + 0.853305 w_k1_MAT_s112
                                 + 0.812781 w_k1_MSFT_s112
                                 + 0.942819 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.866204 w_k1_ADM_s112
                                 + 0.773237 w_k1_AXP_s112
                                 + 0.878715 w_k1_DIS_s112
                                 + 0.927541 w_k1_FDX_s112
                                 + 0.923465 w_k1_IBM_s112
                                 + 1.003 w_k1_INTC_s112 + 1.06311 w_k1_KO_s112
                                 + 0.705181 w_k1_MAT_s112
                                 + 0.9693 w_k1_MSFT_s112
                                 + 0.962221 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.13146 w_k1_ADM_s112
                                 + 1.1942 w_k1_AXP_s112 + 1.51959 w_k1_DIS_s112
                                 + 1.27863 w_k1_FDX_s112 + 1.2846 w_k1_IBM_s112
                                 + 1.30539 w_k1_INTC_s112
                                 + 1.46095 w_k1_KO_s112 + 1.10696 w_k1_MAT_s112
                                 + 1.23639 w_k1_MSFT_s112
                                 + 1.17057 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.00572 w_k1_ADM_s113
                                 + 1.00545 w_k1_AXP_s113
                                 + 0.78868 w_k1_DIS_s113
                                 + 0.950783 w_k1_FDX_s113
                                 + 0.769511 w_k1_IBM_s113
                                 + 1.10424 w_k1_INTC_s113
                                 + 1.00745 w_k1_KO_s113
                                 + 0.923625 w_k1_MAT_s113
                                 + 1.06133 w_k1_MSFT_s113
                                 + 1.04034 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.60625 w_k1_ADM_s113
                                 + 1.55161 w_k1_AXP_s113
                                 + 1.82453 w_k1_DIS_s113
                                 + 1.32292 w_k1_FDX_s113
                                 + 1.81511 w_k1_IBM_s113
                                 + 1.60997 w_k1_INTC_s113
                                 + 1.82432 w_k1_KO_s113 + 1.79286 w_k1_MAT_s113
                                 + 1.47416 w_k1_MSFT_s113
                                 + 1.48798 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.3997 w_k1_ADM_s113
                                 + 0.92516 w_k1_AXP_s113
                                 + 0.784771 w_k1_DIS_s113
                                 + 0.789262 w_k1_FDX_s113
                                 + 1.09148 w_k1_IBM_s113
                                 + 0.793274 w_k1_INTC_s113
                                 + 1.05642 w_k1_KO_s113
                                 + 0.965325 w_k1_MAT_s113
                                 + 0.897719 w_k1_MSFT_s113
                                 + 1.01231 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.31262 w_k1_ADM_s113
                                 + 1.50893 w_k1_AXP_s113
                                 + 1.41149 w_k1_DIS_s113
                                 + 1.23125 w_k1_FDX_s113 + 1.1704 w_k1_IBM_s113
                                 + 1.18355 w_k1_INTC_s113
                                 + 1.29026 w_k1_KO_s113 + 1.33143 w_k1_MAT_s113
                                 + 1.08907 w_k1_MSFT_s113
                                 + 1.13343 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.13382 w_k1_ADM_s113
                                 + 0.551447 w_k1_AXP_s113
                                 + 0.828291 w_k1_DIS_s113
                                 + 0.712172 w_k1_FDX_s113
                                 + 0.87968 w_k1_IBM_s113
                                 + 0.786326 w_k1_INTC_s113
                                 + 0.988598 w_k1_KO_s113
                                 + 0.555617 w_k1_MAT_s113
                                 + 0.794099 w_k1_MSFT_s113
                                 + 0.912861 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.768319 w_k1_ADM_s113
                                 + 0.631592 w_k1_AXP_s113
                                 + 0.621515 w_k1_DIS_s113
                                 + 0.748496 w_k1_FDX_s113
                                 + 0.589374 w_k1_IBM_s113
                                 + 1.03609 w_k1_INTC_s113
                                 + 0.683042 w_k1_KO_s113
                                 + 0.604264 w_k1_MAT_s113
                                 + 0.904633 w_k1_MSFT_s113
                                 + 0.768511 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08144 w_k1_ADM_s113
                                 + 0.906515 w_k1_AXP_s113
                                 + 1.30532 w_k1_DIS_s113
                                 + 0.957721 w_k1_FDX_s113
                                 + 1.45628 w_k1_IBM_s113
                                 + 1.21583 w_k1_INTC_s113
                                 + 1.42363 w_k1_KO_s113 + 1.39049 w_k1_MAT_s113
                                 + 1.0948 w_k1_MSFT_s113 + 1.04995 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.16003 w_k1_ADM_s113
                                 + 0.9045 w_k1_AXP_s113 + 1.08257 w_k1_DIS_s113
                                 + 1.00614 w_k1_FDX_s113
                                 + 0.904274 w_k1_IBM_s113
                                 + 1.03209 w_k1_INTC_s113
                                 + 1.10778 w_k1_KO_s113
                                 + 0.926892 w_k1_MAT_s113
                                 + 1.17008 w_k1_MSFT_s113 + 1.1466 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.808221 w_k1_ADM_s114
                                 + 1.20317 w_k1_AXP_s114
                                 + 1.44862 w_k1_DIS_s114 + 1.1163 w_k1_FDX_s114
                                 + 1.2793 w_k1_IBM_s114
                                 + 1.17372 w_k1_INTC_s114 + 1.3002 w_k1_KO_s114
                                 + 1.59385 w_k1_MAT_s114
                                 + 1.21029 w_k1_MSFT_s114
                                 + 1.19179 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.607595 w_k1_ADM_s114
                                 + 0.567792 w_k1_AXP_s114
                                 + 0.615487 w_k1_DIS_s114
                                 + 0.75097 w_k1_FDX_s114
                                 + 0.566658 w_k1_IBM_s114
                                 + 0.795418 w_k1_INTC_s114
                                 + 0.649561 w_k1_KO_s114
                                 + 0.622624 w_k1_MAT_s114
                                 + 0.749226 w_k1_MSFT_s114
                                 + 0.753725 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.844383 w_k1_ADM_s114
                                 + 0.946682 w_k1_AXP_s114
                                 + 0.795578 w_k1_DIS_s114
                                 + 0.959573 w_k1_FDX_s114
                                 + 0.583903 w_k1_IBM_s114
                                 + 0.90831 w_k1_INTC_s114
                                 + 0.751647 w_k1_KO_s114
                                 + 0.683073 w_k1_MAT_s114
                                 + 0.929496 w_k1_MSFT_s114
                                 + 0.817164 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.12728 w_k1_ADM_s114
                                 + 1.75847 w_k1_AXP_s114
                                 + 1.51337 w_k1_DIS_s114
                                 + 1.51243 w_k1_FDX_s114
                                 + 1.01452 w_k1_IBM_s114
                                 + 1.6331 w_k1_INTC_s114 + 1.22236 w_k1_KO_s114
                                 + 1.11225 w_k1_MAT_s114
                                 + 1.2661 w_k1_MSFT_s114 + 1.08035 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.709361 w_k1_ADM_s114
                                 + 1.0048 w_k1_AXP_s114
                                 + 0.825813 w_k1_DIS_s114
                                 + 0.92311 w_k1_FDX_s114
                                 + 0.881372 w_k1_IBM_s114
                                 + 1.00857 w_k1_INTC_s114
                                 + 0.842701 w_k1_KO_s114
                                 + 1.19596 w_k1_MAT_s114
                                 + 0.875068 w_k1_MSFT_s114
                                 + 0.970687 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.889131 w_k1_ADM_s114
                                 + 0.772729 w_k1_AXP_s114
                                 + 0.910778 w_k1_DIS_s114
                                 + 0.978874 w_k1_FDX_s114
                                 + 0.669442 w_k1_IBM_s114
                                 + 1.15777 w_k1_INTC_s114
                                 + 0.877778 w_k1_KO_s114
                                 + 0.548648 w_k1_MAT_s114
                                 + 0.996121 w_k1_MSFT_s114
                                 + 0.875633 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.729422 w_k1_ADM_s114
                                 + 0.769094 w_k1_AXP_s114
                                 + 0.817581 w_k1_DIS_s114
                                 + 0.806964 w_k1_FDX_s114
                                 + 1.23601 w_k1_IBM_s114
                                 + 1.08622 w_k1_INTC_s114
                                 + 1.05659 w_k1_KO_s114
                                 + 0.938626 w_k1_MAT_s114
                                 + 0.806606 w_k1_MSFT_s114
                                 + 0.959676 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.939494 w_k1_ADM_s114
                                 + 1.02941 w_k1_AXP_s114
                                 + 1.11037 w_k1_DIS_s114
                                 + 1.19873 w_k1_FDX_s114
                                 + 0.639267 w_k1_IBM_s114
                                 + 1.0381 w_k1_INTC_s114
                                 + 0.772836 w_k1_KO_s114
                                 + 0.598017 w_k1_MAT_s114
                                 + 1.04809 w_k1_MSFT_s114
                                 + 0.946166 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.16962 w_k1_ADM_s115
                                 + 1.18502 w_k1_AXP_s115 + 1.3558 w_k1_DIS_s115
                                 + 1.14523 w_k1_FDX_s115
                                 + 1.54342 w_k1_IBM_s115
                                 + 1.2594 w_k1_INTC_s115 + 1.29295 w_k1_KO_s115
                                 + 1.54233 w_k1_MAT_s115
                                 + 1.17049 w_k1_MSFT_s115 + 1.124 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.749733 w_k1_ADM_s115
                                 + 0.912074 w_k1_AXP_s115
                                 + 0.827951 w_k1_DIS_s115
                                 + 0.890506 w_k1_FDX_s115
                                 + 0.974402 w_k1_IBM_s115
                                 + 1.08226 w_k1_INTC_s115
                                 + 0.848903 w_k1_KO_s115
                                 + 1.00338 w_k1_MAT_s115
                                 + 0.886985 w_k1_MSFT_s115 + 0.826 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.09833 w_k1_ADM_s115
                                 + 1.2049 w_k1_AXP_s115 + 1.49456 w_k1_DIS_s115
                                 + 1.29138 w_k1_FDX_s115
                                 + 0.974234 w_k1_IBM_s115
                                 + 1.37733 w_k1_INTC_s115 + 1.0901 w_k1_KO_s115
                                 + 0.592371 w_k1_MAT_s115
                                 + 1.25323 w_k1_MSFT_s115
                                 + 1.06178 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.760143 w_k1_ADM_s115
                                 + 0.70098 w_k1_AXP_s115
                                 + 0.709333 w_k1_DIS_s115
                                 + 0.780852 w_k1_FDX_s115
                                 + 0.717042 w_k1_IBM_s115
                                 + 0.972495 w_k1_INTC_s115
                                 + 0.626208 w_k1_KO_s115
                                 + 0.724693 w_k1_MAT_s115
                                 + 0.892614 w_k1_MSFT_s115
                                 + 0.759238 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.787769 w_k1_ADM_s115
                                 + 0.904163 w_k1_AXP_s115
                                 + 1.07207 w_k1_DIS_s115
                                 + 0.919148 w_k1_FDX_s115
                                 + 1.28682 w_k1_IBM_s115
                                 + 1.18627 w_k1_INTC_s115
                                 + 1.01017 w_k1_KO_s115 + 1.12042 w_k1_MAT_s115
                                 + 1.0691 w_k1_MSFT_s115
                                 + 0.970126 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.01264 w_k1_ADM_s115
                                 + 0.911541 w_k1_AXP_s115
                                 + 0.850436 w_k1_DIS_s115
                                 + 0.986481 w_k1_FDX_s115
                                 + 0.820432 w_k1_IBM_s115
                                 + 0.919372 w_k1_INTC_s115
                                 + 0.794851 w_k1_KO_s115
                                 + 0.76343 w_k1_MAT_s115
                                 + 1.01073 w_k1_MSFT_s115
                                 + 0.975966 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.858817 w_k1_ADM_s115
                                 + 0.79342 w_k1_AXP_s115
                                 + 1.05746 w_k1_DIS_s115
                                 + 0.974486 w_k1_FDX_s115
                                 + 0.860446 w_k1_IBM_s115
                                 + 1.12215 w_k1_INTC_s115
                                 + 0.935121 w_k1_KO_s115
                                 + 0.934761 w_k1_MAT_s115
                                 + 1.1453 w_k1_MSFT_s115 + 1.01629 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.706765 w_k1_ADM_s115
                                 + 0.670165 w_k1_AXP_s115
                                 + 0.619861 w_k1_DIS_s115
                                 + 0.685209 w_k1_FDX_s115
                                 + 0.852675 w_k1_IBM_s115
                                 + 0.685778 w_k1_INTC_s115
                                 + 0.789494 w_k1_KO_s115
                                 + 0.682816 w_k1_MAT_s115
                                 + 0.804494 w_k1_MSFT_s115
                                 + 0.787148 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.87021 w_k1_ADM_s116
                                 + 1.19927 w_k1_AXP_s116
                                 + 0.840176 w_k1_DIS_s116
                                 + 0.898733 w_k1_FDX_s116
                                 + 1.16729 w_k1_IBM_s116
                                 + 0.964666 w_k1_INTC_s116
                                 + 0.935926 w_k1_KO_s116
                                 + 1.43403 w_k1_MAT_s116
                                 + 0.833031 w_k1_MSFT_s116
                                 + 0.911073 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.84802 w_k1_ADM_s116
                                 + 0.834741 w_k1_AXP_s116
                                 + 0.894083 w_k1_DIS_s116
                                 + 0.80666 w_k1_FDX_s116
                                 + 1.07377 w_k1_IBM_s116
                                 + 0.813876 w_k1_INTC_s116
                                 + 0.857592 w_k1_KO_s116
                                 + 1.11277 w_k1_MAT_s116
                                 + 0.74418 w_k1_MSFT_s116
                                 + 0.981553 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.14793 w_k1_ADM_s116
                                 + 1.46017 w_k1_AXP_s116
                                 + 1.44024 w_k1_DIS_s116
                                 + 1.21258 w_k1_FDX_s116
                                 + 1.39484 w_k1_IBM_s116
                                 + 1.10794 w_k1_INTC_s116
                                 + 1.31684 w_k1_KO_s116 + 1.42653 w_k1_MAT_s116
                                 + 1.11592 w_k1_MSFT_s116 + 1.1903 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.673122 w_k1_ADM_s116
                                 + 0.684223 w_k1_AXP_s116
                                 + 0.810136 w_k1_DIS_s116
                                 + 0.849241 w_k1_FDX_s116
                                 + 0.733527 w_k1_IBM_s116
                                 + 0.854091 w_k1_INTC_s116
                                 + 0.785418 w_k1_KO_s116
                                 + 0.927378 w_k1_MAT_s116
                                 + 0.854828 w_k1_MSFT_s116
                                 + 0.902835 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.647722 w_k1_ADM_s116
                                 + 0.607584 w_k1_AXP_s116
                                 + 0.847065 w_k1_DIS_s116
                                 + 0.789236 w_k1_FDX_s116
                                 + 0.852835 w_k1_IBM_s116
                                 + 0.879054 w_k1_INTC_s116
                                 + 0.841107 w_k1_KO_s116
                                 + 0.810785 w_k1_MAT_s116
                                 + 0.854565 w_k1_MSFT_s116
                                 + 0.906726 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.1836 w_k1_ADM_s116
                                 + 1.44134 w_k1_AXP_s116
                                 + 1.11846 w_k1_DIS_s116
                                 + 1.31459 w_k1_FDX_s116
                                 + 0.891873 w_k1_IBM_s116
                                 + 1.15716 w_k1_INTC_s116
                                 + 0.951029 w_k1_KO_s116
                                 + 1.35758 w_k1_MAT_s116
                                 + 1.09469 w_k1_MSFT_s116
                                 + 1.03653 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.06504 w_k1_ADM_s116
                                 + 1.07907 w_k1_AXP_s116
                                 + 1.40953 w_k1_DIS_s116
                                 + 1.11291 w_k1_FDX_s116 + 1.0597 w_k1_IBM_s116
                                 + 1.19103 w_k1_INTC_s116
                                 + 1.21877 w_k1_KO_s116 + 0.92167 w_k1_MAT_s116
                                 + 1.20973 w_k1_MSFT_s116
                                 + 1.18815 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.883532 w_k1_ADM_s116
                                 + 1.0451 w_k1_AXP_s116
                                 + 0.965864 w_k1_DIS_s116
                                 + 1.07308 w_k1_FDX_s116
                                 + 0.847558 w_k1_IBM_s116
                                 + 0.932788 w_k1_INTC_s116
                                 + 0.917671 w_k1_KO_s116
                                 + 0.887937 w_k1_MAT_s116
                                 + 0.960189 w_k1_MSFT_s116
                                 + 0.976089 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.849918 w_k1_ADM_s117
                                 + 0.875552 w_k1_AXP_s117
                                 + 1.09145 w_k1_DIS_s117
                                 + 1.08656 w_k1_FDX_s117
                                 + 0.985723 w_k1_IBM_s117
                                 + 1.19798 w_k1_INTC_s117
                                 + 1.12371 w_k1_KO_s117 + 1.05205 w_k1_MAT_s117
                                 + 1.0667 w_k1_MSFT_s117 + 1.08248 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.643392 w_k1_ADM_s117
                                 + 0.787264 w_k1_AXP_s117
                                 + 0.84391 w_k1_DIS_s117
                                 + 0.844955 w_k1_FDX_s117
                                 + 1.32865 w_k1_IBM_s117
                                 + 0.925802 w_k1_INTC_s117
                                 + 1.00448 w_k1_KO_s117 + 1.10376 w_k1_MAT_s117
                                 + 0.916563 w_k1_MSFT_s117
                                 + 0.957849 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.678467 w_k1_ADM_s117
                                 + 0.77124 w_k1_AXP_s117
                                 + 0.872458 w_k1_DIS_s117
                                 + 0.849634 w_k1_FDX_s117
                                 + 0.916327 w_k1_IBM_s117
                                 + 0.87492 w_k1_INTC_s117
                                 + 0.892972 w_k1_KO_s117
                                 + 0.926841 w_k1_MAT_s117
                                 + 0.979761 w_k1_MSFT_s117
                                 + 0.952208 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.91689 w_k1_ADM_s117
                                 + 0.695212 w_k1_AXP_s117
                                 + 0.887892 w_k1_DIS_s117
                                 + 0.968215 w_k1_FDX_s117
                                 + 0.68564 w_k1_IBM_s117
                                 + 1.0754 w_k1_INTC_s117
                                 + 0.865676 w_k1_KO_s117
                                 + 0.569947 w_k1_MAT_s117
                                 + 1.10661 w_k1_MSFT_s117
                                 + 0.886059 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.84194 w_k1_ADM_s117
                                 + 1.16837 w_k1_AXP_s117
                                 + 1.34952 w_k1_DIS_s117
                                 + 1.10654 w_k1_FDX_s117
                                 + 1.48039 w_k1_IBM_s117
                                 + 1.14239 w_k1_INTC_s117
                                 + 1.29434 w_k1_KO_s117 + 1.76026 w_k1_MAT_s117
                                 + 1.14298 w_k1_MSFT_s117
                                 + 1.13833 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.16757 w_k1_ADM_s117
                                 + 1.42683 w_k1_AXP_s117
                                 + 1.56629 w_k1_DIS_s117
                                 + 1.49046 w_k1_FDX_s117 + 1.1493 w_k1_IBM_s117
                                 + 1.39037 w_k1_INTC_s117
                                 + 1.28511 w_k1_KO_s117 + 1.5336 w_k1_MAT_s117
                                 + 1.39116 w_k1_MSFT_s117
                                 + 1.31539 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.11616 w_k1_ADM_s117
                                 + 1.12969 w_k1_AXP_s117
                                 + 1.36973 w_k1_DIS_s117
                                 + 1.20807 w_k1_FDX_s117
                                 + 0.875779 w_k1_IBM_s117
                                 + 1.15442 w_k1_INTC_s117
                                 + 1.12363 w_k1_KO_s117 + 1.23676 w_k1_MAT_s117
                                 + 1.28861 w_k1_MSFT_s117
                                 + 1.05075 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.927493 w_k1_ADM_s117
                                 + 1.11513 w_k1_AXP_s117
                                 + 1.06511 w_k1_DIS_s117
                                 + 0.963443 w_k1_FDX_s117
                                 + 1.2736 w_k1_IBM_s117 + 1.0273 w_k1_INTC_s117
                                 + 1.07521 w_k1_KO_s117 + 1.54659 w_k1_MAT_s117
                                 + 0.97934 w_k1_MSFT_s117
                                 + 1.00537 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0365330476823971 w_k1_ADM_s000
                                 + 0.0330625249584053 w_k1_ADM_s001
                                 + 0.0311582654277514 w_k1_ADM_s002
                                 + 0.0311558142666494 w_k1_ADM_s003
                                 + 0.0306830331829901 w_k1_ADM_s004
                                 + 0.0223292316692681 w_k1_ADM_s005
                                 + 0.0241780543856609 w_k1_ADM_s006
                                 + 0.0607744770658927 w_k1_ADM_s007
                                 + 0.0103446872104973 w_k1_ADM_s010
                                 + 0.0111174090036843 w_k1_ADM_s011
                                 + 0.0124456703428867 w_k1_ADM_s012
                                 + 0.00987930818798503 w_k1_ADM_s013
                                 + 0.00839573470552792 w_k1_ADM_s014
                                 + 0.00603171917358876 w_k1_ADM_s015
                                 + 0.0128662121768213 w_k1_ADM_s016
                                 + 0.00510732634987182 w_k1_ADM_s017
                                 + 0.0674532809790336 w_k1_ADM_s100
                                 + 0.0412447736034935 w_k1_ADM_s101
                                 + 0.0342849963781133 w_k1_ADM_s102
                                 + 0.0187512957528641 w_k1_ADM_s103
                                 + 0.0485463583473017 w_k1_ADM_s104
                                 + 0.0383260011180258 w_k1_ADM_s105
                                 + 0.0245833529148298 w_k1_ADM_s106
                                 + 0.0124878932665708 w_k1_ADM_s107
                                 + 0.0630282019202745 w_k1_ADM_s110
                                 + 0.0361321718024792 w_k1_ADM_s111
                                 + 0.0622709260399627 w_k1_ADM_s112
                                 + 0.0165320264702682 w_k1_ADM_s113
                                 + 0.0382487114037304 w_k1_ADM_s114
                                 + 0.0436681587379272 w_k1_ADM_s115
                                 + 0.040784116202438 w_k1_ADM_s116
                                 + 0.0462748235309611 w_k1_ADM_s117
                                 + 0.0409349069513597 w_k1_AXP_s000
                                 + 0.0360984416145461 w_k1_AXP_s001
                                 + 0.0288639231298851 w_k1_AXP_s002
                                 + 0.0347888759228698 w_k1_AXP_s003
                                 + 0.0310887017438047 w_k1_AXP_s004
                                 + 0.0219340901110838 w_k1_AXP_s005
                                 + 0.0208075302161695 w_k1_AXP_s006
                                 + 0.0629951141826103 w_k1_AXP_s007
                                 + 0.010084502282541 w_k1_AXP_s010
                                 + 0.0098670918826789 w_k1_AXP_s011
                                 + 0.0126699698514541 w_k1_AXP_s012
                                 + 0.0101473410538703 w_k1_AXP_s013
                                 + 0.00851515887173551 w_k1_AXP_s014
                                 + 0.00646160595537871 w_k1_AXP_s015
                                 + 0.014612934543255 w_k1_AXP_s016
                                 + 0.00632299596294182 w_k1_AXP_s017
                                 + 0.0644986861467042 w_k1_AXP_s100
                                 + 0.0433878965696111 w_k1_AXP_s101
                                 + 0.0416611127232557 w_k1_AXP_s102
                                 + 0.0215647452322359 w_k1_AXP_s103
                                 + 0.0480691767489321 w_k1_AXP_s104
                                 + 0.039012743021398 w_k1_AXP_s105
                                 + 0.023914271037833 w_k1_AXP_s106
                                 + 0.0145091183038487 w_k1_AXP_s107
                                 + 0.0675847089967792 w_k1_AXP_s110
                                 + 0.0357350659662441 w_k1_AXP_s111
                                 + 0.0644586798241249 w_k1_AXP_s112
                                 + 0.014419355803587 w_k1_AXP_s113
                                 + 0.0484851924311655 w_k1_AXP_s114
                                 + 0.0451436250701279 w_k1_AXP_s115
                                 + 0.0460392862681154 w_k1_AXP_s116
                                 + 0.0493063317400901 w_k1_AXP_s117
                                 + 0.0388879147236796 w_k1_DIS_s000
                                 + 0.0343496985909649 w_k1_DIS_s001
                                 + 0.0259169476301109 w_k1_DIS_s002
                                 + 0.0350714299106502 w_k1_DIS_s003
                                 + 0.0334317576722399 w_k1_DIS_s004
                                 + 0.0226196259070645 w_k1_DIS_s005
                                 + 0.0200949437880376 w_k1_DIS_s006
                                 + 0.0609349659649282 w_k1_DIS_s007
                                 + 0.0109260121410021 w_k1_DIS_s010
                                 + 0.0105646407308362 w_k1_DIS_s011
                                 + 0.0133790150200768 w_k1_DIS_s012
                                 + 0.00992495597940231 w_k1_DIS_s013
                                 + 0.0087727730682138 w_k1_DIS_s014
                                 + 0.00660471959021331 w_k1_DIS_s015
                                 + 0.0134737721607393 w_k1_DIS_s016
                                 + 0.006117831112847 w_k1_DIS_s017
                                 + 0.0713262890520936 w_k1_DIS_s100
                                 + 0.0490180693965128 w_k1_DIS_s101
                                 + 0.037107932149437 w_k1_DIS_s102
                                 + 0.0213339418304547 w_k1_DIS_s103
                                 + 0.048928944336838 w_k1_DIS_s104
                                 + 0.035808451180551 w_k1_DIS_s105
                                 + 0.0245182370078319 w_k1_DIS_s106
                                 + 0.014546928936234 w_k1_DIS_s107
                                 + 0.0654764036193442 w_k1_DIS_s110
                                 + 0.038188296389841 w_k1_DIS_s111
                                 + 0.0673576909587631 w_k1_DIS_s112
                                 + 0.0151012306330213 w_k1_DIS_s113
                                 + 0.0495219547249281 w_k1_DIS_s114
                                 + 0.050779113720646 w_k1_DIS_s115
                                 + 0.0479689120361221 w_k1_DIS_s116
                                 + 0.0570460346704906 w_k1_DIS_s117
                                 + 0.0367740635903767 w_k1_FDX_s000
                                 + 0.0336996722176071 w_k1_FDX_s001
                                 + 0.0252579302835066 w_k1_FDX_s002
                                 + 0.0349103978044862 w_k1_FDX_s003
                                 + 0.0298250507886306 w_k1_FDX_s004
                                 + 0.0213441352686848 w_k1_FDX_s005
                                 + 0.0202906833192974 w_k1_FDX_s006
                                 + 0.0613235599775942 w_k1_FDX_s007
                                 + 0.0107031805436754 w_k1_FDX_s010
                                 + 0.0102802415257342 w_k1_FDX_s011
                                 + 0.0125359576919718 w_k1_FDX_s012
                                 + 0.00981590183207307 w_k1_FDX_s013
                                 + 0.00872883119325605 w_k1_FDX_s014
                                 + 0.00641251298932466 w_k1_FDX_s015
                                 + 0.0131688286349782 w_k1_FDX_s016
                                 + 0.00555561853678254 w_k1_FDX_s017
                                 + 0.0703461678165667 w_k1_FDX_s100
                                 + 0.0443144835544768 w_k1_FDX_s101
                                 + 0.0379634161080813 w_k1_FDX_s102
                                 + 0.0210520104420911 w_k1_FDX_s103
                                 + 0.0488385115475347 w_k1_FDX_s104
                                 + 0.0377522410097171 w_k1_FDX_s105
                                 + 0.0244750668844065 w_k1_FDX_s106
                                 + 0.0136699484073943 w_k1_FDX_s107
                                 + 0.0638095897177426 w_k1_FDX_s110
                                 + 0.0358474067242853 w_k1_FDX_s111
                                 + 0.06838511278059 w_k1_FDX_s112
                                 + 0.0139279200672262 w_k1_FDX_s113
                                 + 0.0479185876445342 w_k1_FDX_s114
                                 + 0.0473324970888835 w_k1_FDX_s115
                                 + 0.0445272873697536 w_k1_FDX_s116
                                 + 0.0542876873321155 w_k1_FDX_s117
                                 + 0.0368656262977627 w_k1_IBM_s000
                                 + 0.0358652308738038 w_k1_IBM_s001
                                 + 0.024745232929356 w_k1_IBM_s002
                                 + 0.0347871906242332 w_k1_IBM_s003
                                 + 0.0323384347858099 w_k1_IBM_s004
                                 + 0.0222504993877579 w_k1_IBM_s005
                                 + 0.020178953978367 w_k1_IBM_s006
                                 + 0.0595055493031647 w_k1_IBM_s007
                                 + 0.0109068421999789 w_k1_IBM_s010
                                 + 0.0103453627144082 w_k1_IBM_s011
                                 + 0.012070157761353 w_k1_IBM_s012
                                 + 0.00969463740431344 w_k1_IBM_s013
                                 + 0.00915367680691458 w_k1_IBM_s014
                                 + 0.0066764223905096 w_k1_IBM_s015
                                 + 0.0127961009304552 w_k1_IBM_s016
                                 + 0.00599646076470038 w_k1_IBM_s017
                                 + 0.0719679877479257 w_k1_IBM_s100
                                 + 0.0462903460613544 w_k1_IBM_s101
                                 + 0.0392726211676864 w_k1_IBM_s102
                                 + 0.0225041842889395 w_k1_IBM_s103
                                 + 0.051706175933006 w_k1_IBM_s104
                                 + 0.0391928473751681 w_k1_IBM_s105
                                 + 0.0257477306722389 w_k1_IBM_s106
                                 + 0.0137583483409842 w_k1_IBM_s107
                                 + 0.0634851988820083 w_k1_IBM_s110
                                 + 0.0382934351470863 w_k1_IBM_s111
                                 + 0.0660789261143728 w_k1_IBM_s112
                                 + 0.0146237950149324 w_k1_IBM_s113
                                 + 0.0483142785364551 w_k1_IBM_s114
                                 + 0.0483051403263764 w_k1_IBM_s115
                                 + 0.0455303815633085 w_k1_IBM_s116
                                 + 0.0533127647901551 w_k1_IBM_s117
                                 + 0.0399014246150396 w_k1_INTC_s000
                                 + 0.0291131753948586 w_k1_INTC_s001
                                 + 0.0227936390163266 w_k1_INTC_s002
                                 + 0.0362145101066011 w_k1_INTC_s003
                                 + 0.0278930758950345 w_k1_INTC_s004
                                 + 0.0209490653777735 w_k1_INTC_s005
                                 + 0.0203833925489861 w_k1_INTC_s006
                                 + 0.0644147204779843 w_k1_INTC_s007
                                 + 0.0108837341445909 w_k1_INTC_s010
                                 + 0.010086121043739 w_k1_INTC_s011
                                 + 0.0130510620572823 w_k1_INTC_s012
                                 + 0.00962919951125536 w_k1_INTC_s013
                                 + 0.010106945012425 w_k1_INTC_s014
                                 + 0.00795000531228574 w_k1_INTC_s015
                                 + 0.0155586038171387 w_k1_INTC_s016
                                 + 0.00570286948072726 w_k1_INTC_s017
                                 + 0.0628263075511876 w_k1_INTC_s100
                                 + 0.0442821666942301 w_k1_INTC_s101
                                 + 0.0388616581734757 w_k1_INTC_s102
                                 + 0.0224244619663899 w_k1_INTC_s103
                                 + 0.0499854665412057 w_k1_INTC_s104
                                 + 0.0441427150216997 w_k1_INTC_s105
                                 + 0.0237151485480158 w_k1_INTC_s106
                                 + 0.0127106776495424 w_k1_INTC_s107
                                 + 0.0616613143416258 w_k1_INTC_s110
                                 + 0.0327992107659277 w_k1_INTC_s111
                                 + 0.0730248265009848 w_k1_INTC_s112
                                 + 0.0153795726265433 w_k1_INTC_s113
                                 + 0.0531001962038805 w_k1_INTC_s114
                                 + 0.0526269488378714 w_k1_INTC_s115
                                 + 0.0438485619564281 w_k1_INTC_s116
                                 + 0.0563718009272968 w_k1_INTC_s117
                                 + 0.0361192636660223 w_k1_KO_s000
                                 + 0.0282692406867652 w_k1_KO_s001
                                 + 0.0250184845159546 w_k1_KO_s002
                                 + 0.037369081928298 w_k1_KO_s003
                                 + 0.030337219313857 w_k1_KO_s004
                                 + 0.022318577977476 w_k1_KO_s005
                                 + 0.0220142126476631 w_k1_KO_s006
                                 + 0.0626573759094793 w_k1_KO_s007
                                 + 0.0114889204179167 w_k1_KO_s010
                                 + 0.0106216747521511 w_k1_KO_s011
                                 + 0.0134783412635789 w_k1_KO_s012
                                 + 0.00946577176919423 w_k1_KO_s013
                                 + 0.00883707613432209 w_k1_KO_s014
                                 + 0.00591205367930721 w_k1_KO_s015
                                 + 0.0131520333554113 w_k1_KO_s016
                                 + 0.00595435354906617 w_k1_KO_s017
                                 + 0.0640130100356052 w_k1_KO_s100
                                 + 0.0464276764849893 w_k1_KO_s101
                                 + 0.0375168240307532 w_k1_KO_s102
                                 + 0.0218500042664946 w_k1_KO_s103
                                 + 0.0468603275766722 w_k1_KO_s104
                                 + 0.0398746169083773 w_k1_KO_s105
                                 + 0.0230757431156324 w_k1_KO_s106
                                 + 0.0122042132205838 w_k1_KO_s107
                                 + 0.0631016937598722 w_k1_KO_s110
                                 + 0.0322266363835203 w_k1_KO_s111
                                 + 0.0689742012476879 w_k1_KO_s112
                                 + 0.0162440721303722 w_k1_KO_s113
                                 + 0.0484938992820213 w_k1_KO_s114
                                 + 0.0449648885461429 w_k1_KO_s115
                                 + 0.0447831978877 w_k1_KO_s116
                                 + 0.0547173291324118 w_k1_KO_s117
                                 + 0.0399859590194361 w_k1_MAT_s000
                                 + 0.0357296876000886 w_k1_MAT_s001
                                 + 0.0276282418093628 w_k1_MAT_s002
                                 + 0.0385461766888664 w_k1_MAT_s003
                                 + 0.0264156654735068 w_k1_MAT_s004
                                 + 0.0208929179357638 w_k1_MAT_s005
                                 + 0.019847753639139 w_k1_MAT_s006
                                 + 0.0734897524720227 w_k1_MAT_s007
                                 + 0.0106930421608987 w_k1_MAT_s010
                                 + 0.00991554962543031 w_k1_MAT_s011
                                 + 0.0121826716973388 w_k1_MAT_s012
                                 + 0.0111423358957172 w_k1_MAT_s013
                                 + 0.00738680534700301 w_k1_MAT_s014
                                 + 0.0052104764187941 w_k1_MAT_s015
                                 + 0.0126435962713187 w_k1_MAT_s016
                                 + 0.00646083547409192 w_k1_MAT_s017
                                 + 0.0684988398970922 w_k1_MAT_s100
                                 + 0.0449210566415131 w_k1_MAT_s101
                                 + 0.0405224922651798 w_k1_MAT_s102
                                 + 0.0240208669165303 w_k1_MAT_s103
                                 + 0.0462810608325678 w_k1_MAT_s104
                                 + 0.0356573780486592 w_k1_MAT_s105
                                 + 0.0274834117413946 w_k1_MAT_s106
                                 + 0.0145809324672867 w_k1_MAT_s107
                                 + 0.0766216937765732 w_k1_MAT_s110
                                 + 0.0416295487349462 w_k1_MAT_s111
                                 + 0.0563926338699886 w_k1_MAT_s112
                                 + 0.0148128950169492 w_k1_MAT_s113
                                 + 0.0515471187482837 w_k1_MAT_s114
                                 + 0.0427319364850511 w_k1_MAT_s115
                                 + 0.0482728292457937 w_k1_MAT_s116
                                 + 0.0583866063297387 w_k1_MAT_s117
                                 + 0.0388420898925099 w_k1_MSFT_s000
                                 + 0.031075333982797 w_k1_MSFT_s001
                                 + 0.0242549180866439 w_k1_MSFT_s002
                                 + 0.0322935188422371 w_k1_MSFT_s003
                                 + 0.0278391339888903 w_k1_MSFT_s004
                                 + 0.0212980154189833 w_k1_MSFT_s005
                                 + 0.0198555578370936 w_k1_MSFT_s006
                                 + 0.064109966618528 w_k1_MSFT_s007
                                 + 0.0117762451831163 w_k1_MSFT_s010
                                 + 0.0103189282962321 w_k1_MSFT_s011
                                 + 0.0134097933677057 w_k1_MSFT_s012
                                 + 0.00933826494014547 w_k1_MSFT_s013
                                 + 0.00894483904055914 w_k1_MSFT_s014
                                 + 0.00689417009332277 w_k1_MSFT_s015
                                 + 0.0154302734158888 w_k1_MSFT_s016
                                 + 0.00508422347550001 w_k1_MSFT_s017
                                 + 0.0689704910130361 w_k1_MSFT_s100
                                 + 0.0479611347278211 w_k1_MSFT_s101
                                 + 0.0344063266131367 w_k1_MSFT_s102
                                 + 0.0213555192588583 w_k1_MSFT_s103
                                 + 0.0450140914466901 w_k1_MSFT_s104
                                 + 0.0395854461588628 w_k1_MSFT_s105
                                 + 0.025462481087768 w_k1_MSFT_s106
                                 + 0.0139769006449841 w_k1_MSFT_s107
                                 + 0.0636805781396195 w_k1_MSFT_s110
                                 + 0.0368988347508832 w_k1_MSFT_s111
                                 + 0.0672051151368955 w_k1_MSFT_s112
                                 + 0.0151010820976723 w_k1_MSFT_s113
                                 + 0.0469889053900454 w_k1_MSFT_s114
                                 + 0.0496798643060024 w_k1_MSFT_s115
                                 + 0.0430496436437049 w_k1_MSFT_s116
                                 + 0.0581352020496242 w_k1_MSFT_s117
                                 + 0.0381954498071643 w_k1_PG_s000
                                 + 0.0330714586244956 w_k1_PG_s001
                                 + 0.0254939688423458 w_k1_PG_s002
                                 + 0.0351559709836742 w_k1_PG_s003
                                 + 0.0296154259882272 w_k1_PG_s004
                                 + 0.0223020469538935 w_k1_PG_s005
                                 + 0.0217507158582915 w_k1_PG_s006
                                 + 0.0610019919003851 w_k1_PG_s007
                                 + 0.0113049243691977 w_k1_PG_s010
                                 + 0.00962276967802489 w_k1_PG_s011
                                 + 0.0124910370137603 w_k1_PG_s012
                                 + 0.00934268216302761 w_k1_PG_s013
                                 + 0.00788866536717213 w_k1_PG_s014
                                 + 0.00619350475172519 w_k1_PG_s015
                                 + 0.0123741373764925 w_k1_PG_s016
                                 + 0.00550184050113148 w_k1_PG_s017
                                 + 0.0676534021534441 w_k1_PG_s100
                                 + 0.041686437023903 w_k1_PG_s101
                                 + 0.0374846691325725 w_k1_PG_s102
                                 + 0.0197664258529307 w_k1_PG_s103
                                 + 0.0485858461305538 w_k1_PG_s104
                                 + 0.0388091320620491 w_k1_PG_s105
                                 + 0.0256717918466382 w_k1_PG_s106
                                 + 0.0135003117194285 w_k1_PG_s107
                                 + 0.0687945036350799 w_k1_PG_s110
                                 + 0.0391639899514199 w_k1_PG_s111
                                 + 0.0655113340321853 w_k1_PG_s112
                                 + 0.015144473484043 w_k1_PG_s113
                                 + 0.0478703617680604 w_k1_PG_s114
                                 + 0.0450337534021915 w_k1_PG_s115
                                 + 0.0457034017119861 w_k1_PG_s116
                                 + 0.0535607250421134 w_k1_PG_s117 >= 1.209
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
