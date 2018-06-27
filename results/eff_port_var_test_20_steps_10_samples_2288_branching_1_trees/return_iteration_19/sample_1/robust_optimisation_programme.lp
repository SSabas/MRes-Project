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
 w_k1_ADM_s0_return_constraint:  1.04996 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.969814 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.00928 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.32513 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.99881 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.911893 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.908057 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.2546 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.22403 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.870708 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.763503 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.12978 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.00699 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.870349 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.755467 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.05644 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.696152 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.97743 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.03299 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.962733 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.886974 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.757183 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.20271 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.49115 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.3035 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.967215 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.10196 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.893521 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.612775 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.905812 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.09096 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.953194 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10566 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.895097 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.694755 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.37966 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.26441 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.704102 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20985 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.87949 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.851472 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.29938 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.962912 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.01065 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.787856 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.4997 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.10508 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.876058 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.584808 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.43187 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.28891 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.961366 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.710462 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.12349 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.711664 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.919144 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.35474 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.10848 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.988874 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.804295 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.77406 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.9759 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.37931 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.00878 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.04448 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.763932 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.587721 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.854839 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.93316 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.911625 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.22409 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.14083 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.624828 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.64905 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.15412 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.642743 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31694 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.857182 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.829471 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.30605 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.02961 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.0543 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.672777 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.52013 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.30778 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.799109 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.529434 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.58773 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.07974 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.924943 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.71414 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.02797 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.63345 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.22872 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.51545 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.06305 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.867853 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.638946 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.40014 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 2.37724 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.32817 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.841803 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.26282 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.646797 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.510147 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.795176 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.978636 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.825446 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.203 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.796264 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.568966 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.71532 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.15693 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.540162 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17972 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.914115 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.871669 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.13594 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.06525 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.946583 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.794678 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.44558 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.2789 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.839103 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.709143 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.23081 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.10515 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.941137 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.950986 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.27049 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.670537 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.13844 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.38205 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.93985 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.972144 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.809954 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.24253 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.60829 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.3025 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.926546 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.10028 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.784212 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.59801 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.789916 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.970487 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.881016 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.997566 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.957902 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.728516 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.45026 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.14483 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.638234 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20929 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.900415 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.901983 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.15856 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.877643 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.37638 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.811022 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.13278 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.791108 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.995838 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.596785 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.81376 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.991927 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.22991 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.547476 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.617315 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.33717 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.776154 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.15444 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.51747 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.878071 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.866043 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.6685 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.54096 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.885335 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.84261 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.13689 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.788783 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.71121 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.24349 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.931167 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.11209 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.66 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.944809 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.611444 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.37367 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.788967 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.794707 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16267 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.890265 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.918876 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.08859 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13006 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1132 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.841027 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.13108 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.16818 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.950945 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.663644 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.19197 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.951185 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.878582 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.850294 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.04534 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.800437 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.974335 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.31107 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.997918 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.92236 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.927411 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.13859 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.34397 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.11359 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.889256 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.04467 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.849625 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.742434 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.848201 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.884654 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.907853 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.931559 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.85163 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.66536 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.25478 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.03614 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.569488 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26574 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.907644 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.749864 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.07971 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.87946 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23048 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.922142 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.37281 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.1072 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.05099 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.729058 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.5085 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.07385 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.12454 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.750089 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.870178 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.935366 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.03688 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.4189 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.21449 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.892167 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.780833 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.48969 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.85636 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.22796 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.05263 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.26181 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.818362 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.71252 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.992233 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.950761 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.959998 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.23069 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.795605 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.547388 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.24869 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.915003 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.642599 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25381 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.813171 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.874904 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.1354 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.857248 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.34504 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.678912 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.53209 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.661282 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.747963 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.587761 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.88722 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.02346 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.17038 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.638737 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.79788 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.26421 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.895101 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.51885 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.78687 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.807922 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.88789 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.88802 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.51214 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.851097 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.13057 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.37551 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.743045 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.572034 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.39479 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.889762 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.04143 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.74145 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.18908 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.521712 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.80711 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.733341 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.900586 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14373 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.933328 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.895649 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.14591 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.02427 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.938974 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.792462 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.35152 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.26853 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.857869 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.738169 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.2 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.04111 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.997668 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.950286 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.13577 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.795234 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.13775 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.33584 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.964127 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.861814 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.817957 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.25628 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.51809 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.18623 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.919615 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.13007 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.852745 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.709272 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.877248 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.03576 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.894697 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.11614 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.937483 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.714653 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.43483 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.12431 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.678866 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13278 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.916832 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.877531 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.1428 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.997432 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.07663 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.880567 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.36264 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.10168 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.931816 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.804928 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.37459 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05797 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08314 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.799036 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04911 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.850038 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.04876 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.10008 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.15364 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.870983 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.83739 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.24692 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.50589 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.05835 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.961464 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.06075 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.844318 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.749434 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.887385 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.919981 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.935781 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.12858 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.957188 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.730532 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.31486 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.05341 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.734715 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.2290009046378 z_k1_s0000
                                 - 0.308999995229056 z_k1_s0001
                                 - 0.522000975148177 z_k1_s0002
                                 - 0.444001442082518 z_k1_s0003
                                 - 0.269000849685033 z_k1_s0004
                                 - 0.152000484082265 z_k1_s0005
                                 - 0.237001000247675 z_k1_s0006
                                 - 0.503002113892708 z_k1_s0007
                                 - 0.168000365834519 z_k1_s0010
                                 - 0.152000330993137 z_k1_s0011
                                 - 0.145999831426923 z_k1_s0012
                                 - 0.0470000536964925 z_k1_s0013
                                 - 0.0850001364447766 z_k1_s0014
                                 - 0.1069999897514 z_k1_s0015
                                 - 0.188000409386248 z_k1_s0016
                                 - 0.080000076906775 z_k1_s0017
                                 - 0.251999966663938 z_k1_s0020
                                 - 0.251999966663938 z_k1_s0021
                                 - 0.37699901507014 z_k1_s0022
                                 - 0.319000112288631 z_k1_s0023
                                 - 0.184999616139568 z_k1_s0024
                                 - 0.160999716885706 z_k1_s0025
                                 - 0.144999715749905 z_k1_s0026
                                 - 0.357998667953171 z_k1_s0027
                                 - 0.345000454419186 z_k1_s0030
                                 - 0.537001922393489 z_k1_s0031
                                 - 0.516001865802711 z_k1_s0032
                                 - 0.298000517878216 z_k1_s0033
                                 - 0.800999782101701 z_k1_s0034
                                 - 0.169000217777797 z_k1_s0035
                                 - 0.46000171489397 z_k1_s0036
                                 - 0.201000351540009 z_k1_s0037
                                 - 0.228999525182934 z_k1_s0040
                                 - 0.13000009674162 z_k1_s0041
                                 - 0.279999993289516 z_k1_s0042
                                 - 0.227000384002802 z_k1_s0043
                                 - 0.121000047028713 z_k1_s0044
                                 - 0.158000655270902 z_k1_s0045
                                 - 0.166000015993082 z_k1_s0046
                                 - 0.0870001077578788 z_k1_s0047
                                 - 0.13500008397425 z_k1_s0050
                                 - 0.411000061525861 z_k1_s0051
                                 - 0.106000158825056 z_k1_s0052
                                 - 0.0890001346823597 z_k1_s0053
                                 - 0.156000403886096 z_k1_s0054
                                 - 0.0910000712167305 z_k1_s0055
                                 - 0.208999990148147 z_k1_s0056
                                 - 0.211999754049567 z_k1_s0057
                                 - 0.777999395552583 z_k1_s0060
                                 - 0.383001453079549 z_k1_s0061
                                 - 0.283999725200202 z_k1_s0062
                                 - 0.234999713784706 z_k1_s0063
                                 - 0.630997908506897 z_k1_s0064
                                 - 0.430998306647488 z_k1_s0065
                                 - 0.695999776710105 z_k1_s0066
                                 - 0.193999904363467 z_k1_s0067
                                 - 0.58100154354765 z_k1_s0070
                                 - 0.903000097316516 z_k1_s0071
                                 - 0.839999757317034 z_k1_s0072
                                 - 0.214000563603428 z_k1_s0073
                                 - 0.33100085388751 z_k1_s0074
                                 - 0.650000551353921 z_k1_s0075
                                 - 0.528002167232416 z_k1_s0076
                                 - 0.729000522844441 z_k1_s0077
                                 - 0.274999091383863 z_k1_s0100
                                 - 0.365999574234065 z_k1_s0101
                                 - 0.152999902838783 z_k1_s0102
                                 - 0.294999257702554 z_k1_s0103
                                 - 0.113999852417177 z_k1_s0104
                                 - 0.340999822835435 z_k1_s0105
                                 - 0.182999787117033 z_k1_s0106
                                 - 0.0990000015779987 z_k1_s0107
                                 - 0.0889999201935014 z_k1_s0110
                                 - 0.26900026162741 z_k1_s0111
                                 - 0.285999890927391 z_k1_s0112
                                 - 0.170999372722476 z_k1_s0113
                                 - 0.115999997128581 z_k1_s0114
                                 - 0.155000251145878 z_k1_s0115
                                 - 0.109999908587472 z_k1_s0116
                                 - 0.0899999135170255 z_k1_s0117
                                 - 0.265999718285853 z_k1_s0120
                                 - 0.0779999732657583 z_k1_s0121
                                 - 0.291000390229787 z_k1_s0122
                                 - 0.107999983398736 z_k1_s0123
                                 - 0.063999995077027 z_k1_s0124
                                 - 0.080000027021273 z_k1_s0125
                                 - 0.2899998325522 z_k1_s0126
                                 - 0.150000050664887 z_k1_s0127
                                 - 0.163999105789912 z_k1_s0130
                                 - 0.16699952835903 z_k1_s0131
                                 - 0.159999905697104 z_k1_s0132
                                 - 0.266000385146177 z_k1_s0133
                                 - 0.214999745468044 z_k1_s0134
                                 - 0.348999533564432 z_k1_s0135
                                 - 0.197000482051774 z_k1_s0136
                                 - 0.117999879326638 z_k1_s0137
                                 - 0.459000439602086 z_k1_s0140
                                 - 0.224999646852843 z_k1_s0141
                                 - 0.333999416552309 z_k1_s0142
                                 - 0.20699970645657 z_k1_s0143
                                 - 0.406999305459226 z_k1_s0144
                                 - 0.799000620085328 z_k1_s0145
                                 - 0.722000221891122 z_k1_s0146
                                 - 0.765998769861722 z_k1_s0147
                                 - 0.720000034833037 z_k1_s0150
                                 - 0.425999537335154 z_k1_s0151
                                 - 0.403999697362275 z_k1_s0152
                                 - 0.262999709128282 z_k1_s0153
                                 - 0.723997085813591 z_k1_s0154
                                 - 1.07199925879685 z_k1_s0155
                                 - 0.277999802791354 z_k1_s0156
                                 - 0.574000610177273 z_k1_s0157
                                 - 0.247999681557415 z_k1_s0160
                                 - 0.436999555974802 z_k1_s0161
                                 - 0.689999157381458 z_k1_s0162
                                 - 0.257999521255897 z_k1_s0163
                                 - 0.355998918339713 z_k1_s0164
                                 - 0.154999666523566 z_k1_s0165
                                 - 0.396000428330736 z_k1_s0166
                                 - 0.148999708924549 z_k1_s0167
                                 - 0.476998204345373 z_k1_s0170
                                 - 0.482997733269382 z_k1_s0171
                                 - 0.234999218500851 z_k1_s0172
                                 - 0.234999218500851 z_k1_s0173
                                 - 0.172999411534286 z_k1_s0174
                                 - 0.322999994995218 z_k1_s0175
                                 - 0.218999206974191 z_k1_s0176
                                 - 0.231999216339603 z_k1_s0177
                                 - 0.329999160418647 z_k1_s1000
                                 - 0.166999694163206 z_k1_s1001
                                 - 0.55499795025045 z_k1_s1002
                                 - 0.140999896360332 z_k1_s1003
                                 - 0.461998824586106 z_k1_s1004
                                 - 0.535998551286166 z_k1_s1005
                                 - 0.451998254816691 z_k1_s1006
                                 - 0.162999746855041 z_k1_s1007
                                 - 0.515998448837788 z_k1_s1010
                                 - 0.327999303408109 z_k1_s1011
                                 - 0.15199980659138 z_k1_s1012
                                 - 0.200999567134246 z_k1_s1013
                                 - 0.220999308988702 z_k1_s1014
                                 - 0.124999762088417 z_k1_s1015
                                 - 0.315999065295988 z_k1_s1016
                                 - 0.10599981082696 z_k1_s1017
                                 - 0.565998808637308 z_k1_s1020
                                 - 0.60399850227795 z_k1_s1021
                                 - 0.626997409063176 z_k1_s1022
                                 - 0.813999662283438 z_k1_s1023
                                 - 0.643998957897337 z_k1_s1024
                                 - 0.402998799982747 z_k1_s1025
                                 - 0.233999462183052 z_k1_s1026
                                 - 1.03599553819106 z_k1_s1027
                                 - 0.904997813658537 z_k1_s1030
                                 - 0.62099927904024 z_k1_s1031
                                 - 2.37500119882239 z_k1_s1032
                                 - 1.279996779263 z_k1_s1033
                                 - 0.559999850301608 z_k1_s1034
                                 - 0.640999688872125 z_k1_s1035
                                 - 0.616999507073627 z_k1_s1036
                                 - 0.750999230449552 z_k1_s1037
                                 - 2.03000118704774 z_k1_s1040
                                 - 0.441000064469101 z_k1_s1041
                                 - 0.928999042030082 z_k1_s1042
                                 - 1.82199895636423 z_k1_s1043
                                 - 0.732999636576797 z_k1_s1044
                                 - 1.18299808077067 z_k1_s1045
                                 - 1.41599684935574 z_k1_s1046
                                 - 0.793999976171657 z_k1_s1047
                                 - 0.942000744189765 z_k1_s1050
                                 - 1.09100005424071 z_k1_s1051
                                 - 0.642999945399284 z_k1_s1052
                                 - 0.366000379378978 z_k1_s1053
                                 - 0.726999309930947 z_k1_s1054
                                 - 0.26600026138743 z_k1_s1055
                                 - 0.483000806014575 z_k1_s1056
                                 - 0.729003667308116 z_k1_s1057
                                 - 0.44299849228406 z_k1_s1060
                                 - 0.238998663531941 z_k1_s1061
                                 - 0.510996347712779 z_k1_s1062
                                 - 0.294998469170578 z_k1_s1063
                                 - 0.538997920523021 z_k1_s1064
                                 - 0.76399579764984 z_k1_s1065
                                 - 0.942993774788103 z_k1_s1066
                                 - 0.440998678153136 z_k1_s1067
                                 - 1.01699768333346 z_k1_s1070
                                 - 0.958997800834644 z_k1_s1071
                                 - 0.339998849780725 z_k1_s1072
                                 - 0.501996590462897 z_k1_s1073
                                 - 0.74399729647136 z_k1_s1074
                                 - 0.214999513166454 z_k1_s1075
                                 - 0.613996876252069 z_k1_s1076
                                 - 0.564997060968315 z_k1_s1077
                                 - 0.51800217721642 z_k1_s1100
                                 - 0.417002380841713 z_k1_s1101
                                 - 0.211001128944678 z_k1_s1102
                                 - 0.182001034563984 z_k1_s1103
                                 - 0.187000911611627 z_k1_s1104
                                 - 0.378002294864451 z_k1_s1105
                                 - 0.120000611662685 z_k1_s1106
                                 - 0.36200183330737 z_k1_s1107
                                 - 0.490998731724831 z_k1_s1110
                                 - 0.608998252893053 z_k1_s1111
                                 - 0.829999123028758 z_k1_s1112
                                 - 0.229999773456149 z_k1_s1113
                                 - 0.275999864867271 z_k1_s1114
                                 - 0.441000267059282 z_k1_s1115
                                 - 0.175999859338593 z_k1_s1116
                                 - 0.36500109684883 z_k1_s1117
                                 - 0.413001200154964 z_k1_s1120
                                 - 0.245999952691348 z_k1_s1121
                                 - 0.470000291825655 z_k1_s1122
                                 - 0.165000372083984 z_k1_s1123
                                 - 0.178000236640593 z_k1_s1124
                                 - 0.15300022981584 z_k1_s1125
                                 - 0.26700004836031 z_k1_s1126
                                 - 0.152000303126989 z_k1_s1127
                                 - 0.105000537820752 z_k1_s1130
                                 - 0.157001030257136 z_k1_s1131
                                 - 0.14900093965145 z_k1_s1132
                                 - 0.320001694215952 z_k1_s1133
                                 - 0.148000807700021 z_k1_s1134
                                 - 0.497003047487877 z_k1_s1135
                                 - 0.29600200140234 z_k1_s1136
                                 - 0.258002005277916 z_k1_s1137
                                 - 0.219001047838856 z_k1_s1140
                                 - 0.43600034704132 z_k1_s1141
                                 - 0.125000267576372 z_k1_s1142
                                 - 0.342001191281788 z_k1_s1143
                                 - 0.135000166242259 z_k1_s1144
                                 - 0.23500059690375 z_k1_s1145
                                 - 0.217001104205744 z_k1_s1146
                                 - 0.0960001823946117 z_k1_s1147
                                 - 0.174000383528209 z_k1_s1150
                                 - 0.221000582007384 z_k1_s1151
                                 - 0.778001668309897 z_k1_s1152
                                 - 0.710001009998464 z_k1_s1153
                                 - 0.217000598253939 z_k1_s1154
                                 - 0.313002388843305 z_k1_s1155
                                 - 0.35000247218839 z_k1_s1156
                                 - 0.352002308314635 z_k1_s1157
                                 - 0.460999728741113 z_k1_s1160
                                 - 0.298999438614637 z_k1_s1161
                                 - 0.340999280869984 z_k1_s1162
                                 - 0.3599981986179 z_k1_s1163
                                 - 0.752999979895399 z_k1_s1164
                                 - 0.247999562768956 z_k1_s1165
                                 - 0.4140009161563 z_k1_s1166
                                 - 0.268999641146404 z_k1_s1167
                                 - 0.0980000954561737 z_k1_s1170
                                 - 0.22700064601121 z_k1_s1171
                                 - 0.146999832433909 z_k1_s1172
                                 - 0.12600056665844 z_k1_s1173
                                 - 0.178999660664266 z_k1_s1174
                                 - 0.196000190912347 z_k1_s1175
                                 - 0.0990000900883724 z_k1_s1176
                                 - 0.170999703606677 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.04297 w_k1_ADM_s000
                                 + 1.50513 w_k1_AXP_s000
                                 + 1.76298 w_k1_DIS_s000
                                 + 1.35824 w_k1_FDX_s000
                                 + 1.05134 w_k1_IBM_s000
                                 + 1.33827 w_k1_INTC_s000
                                 + 1.19185 w_k1_KO_s000 + 1.14583 w_k1_MAT_s000
                                 + 1.3078 w_k1_MSFT_s000 + 1.11058 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.749175 w_k1_ADM_s000
                                 + 0.658812 w_k1_AXP_s000
                                 + 0.678117 w_k1_DIS_s000
                                 + 0.801715 w_k1_FDX_s000
                                 + 0.750595 w_k1_IBM_s000
                                 + 0.626528 w_k1_INTC_s000
                                 + 0.682193 w_k1_KO_s000 + 0.8935 w_k1_MAT_s000
                                 + 0.738314 w_k1_MSFT_s000
                                 + 0.832331 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.815566 w_k1_ADM_s000
                                 + 0.90303 w_k1_AXP_s000
                                 + 0.820632 w_k1_DIS_s000
                                 + 0.742581 w_k1_FDX_s000 + 1.337 w_k1_IBM_s000
                                 + 0.912326 w_k1_INTC_s000
                                 + 0.963597 w_k1_KO_s000
                                 + 1.54578 w_k1_MAT_s000
                                 + 0.975594 w_k1_MSFT_s000
                                 + 0.855491 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.00348 w_k1_ADM_s000
                                 + 0.959891 w_k1_AXP_s000
                                 + 0.815896 w_k1_DIS_s000
                                 + 1.00781 w_k1_FDX_s000
                                 + 0.803741 w_k1_IBM_s000
                                 + 0.832984 w_k1_INTC_s000
                                 + 0.873161 w_k1_KO_s000
                                 + 1.12226 w_k1_MAT_s000
                                 + 0.905333 w_k1_MSFT_s000
                                 + 0.979342 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.13166 w_k1_ADM_s000
                                 + 0.877087 w_k1_AXP_s000
                                 + 1.1596 w_k1_DIS_s000 + 1.23988 w_k1_FDX_s000
                                 + 0.657465 w_k1_IBM_s000
                                 + 1.05834 w_k1_INTC_s000
                                 + 0.967233 w_k1_KO_s000
                                 + 0.775526 w_k1_MAT_s000
                                 + 1.28354 w_k1_MSFT_s000
                                 + 1.03119 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.80054 w_k1_ADM_s000
                                 + 1.56167 w_k1_AXP_s000
                                 + 1.40978 w_k1_DIS_s000
                                 + 1.36916 w_k1_FDX_s000
                                 + 1.18389 w_k1_IBM_s000
                                 + 1.19947 w_k1_INTC_s000 + 1.1927 w_k1_KO_s000
                                 + 1.66953 w_k1_MAT_s000
                                 + 1.29043 w_k1_MSFT_s000
                                 + 1.18776 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.939759 w_k1_ADM_s000
                                 + 1.02563 w_k1_AXP_s000
                                 + 1.05798 w_k1_DIS_s000
                                 + 1.03827 w_k1_FDX_s000
                                 + 0.849444 w_k1_IBM_s000
                                 + 1.05605 w_k1_INTC_s000
                                 + 0.986238 w_k1_KO_s000
                                 + 0.893256 w_k1_MAT_s000
                                 + 1.00749 w_k1_MSFT_s000
                                 + 1.07816 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.902741 w_k1_ADM_s000
                                 + 1.08756 w_k1_AXP_s000
                                 + 1.24859 w_k1_DIS_s000
                                 + 1.03729 w_k1_FDX_s000
                                 + 1.16078 w_k1_IBM_s000
                                 + 1.04518 w_k1_INTC_s000
                                 + 1.14608 w_k1_KO_s000 + 1.56469 w_k1_MAT_s000
                                 + 1.09042 w_k1_MSFT_s000
                                 + 1.14914 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.05153 w_k1_ADM_s001
                                 + 1.13211 w_k1_AXP_s001
                                 + 0.962565 w_k1_DIS_s001
                                 + 0.938275 w_k1_FDX_s001
                                 + 1.2128 w_k1_IBM_s001
                                 + 1.18061 w_k1_INTC_s001
                                 + 1.29833 w_k1_KO_s001
                                 + 0.975981 w_k1_MAT_s001
                                 + 1.0339 w_k1_MSFT_s001
                                 + 0.896346 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.07353 w_k1_ADM_s001
                                 + 1.09482 w_k1_AXP_s001
                                 + 1.19512 w_k1_DIS_s001
                                 + 1.07944 w_k1_FDX_s001
                                 + 1.06491 w_k1_IBM_s001
                                 + 1.28996 w_k1_INTC_s001
                                 + 1.21438 w_k1_KO_s001
                                 + 0.885062 w_k1_MAT_s001
                                 + 1.1603 w_k1_MSFT_s001
                                 + 0.980174 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.908604 w_k1_ADM_s001
                                 + 0.931542 w_k1_AXP_s001
                                 + 0.875799 w_k1_DIS_s001
                                 + 0.948764 w_k1_FDX_s001
                                 + 0.705774 w_k1_IBM_s001
                                 + 0.927306 w_k1_INTC_s001
                                 + 0.810353 w_k1_KO_s001
                                 + 0.604805 w_k1_MAT_s001
                                 + 0.948324 w_k1_MSFT_s001
                                 + 0.891562 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.775436 w_k1_ADM_s001
                                 + 0.930191 w_k1_AXP_s001
                                 + 1.07127 w_k1_DIS_s001
                                 + 0.848617 w_k1_FDX_s001
                                 + 1.44823 w_k1_IBM_s001
                                 + 1.02865 w_k1_INTC_s001
                                 + 1.21585 w_k1_KO_s001 + 1.19945 w_k1_MAT_s001
                                 + 0.880016 w_k1_MSFT_s001
                                 + 0.907521 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.3868 w_k1_ADM_s001
                                 + 1.36475 w_k1_AXP_s001
                                 + 1.15414 w_k1_DIS_s001
                                 + 1.30124 w_k1_FDX_s001
                                 + 0.705902 w_k1_IBM_s001
                                 + 1.31067 w_k1_INTC_s001
                                 + 0.974536 w_k1_KO_s001
                                 + 0.869278 w_k1_MAT_s001
                                 + 1.35578 w_k1_MSFT_s001 + 1.0309 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.12736 w_k1_ADM_s001
                                 + 0.959148 w_k1_AXP_s001
                                 + 1.0012 w_k1_DIS_s001
                                 + 0.936739 w_k1_FDX_s001
                                 + 0.831649 w_k1_IBM_s001
                                 + 1.00407 w_k1_INTC_s001
                                 + 0.874136 w_k1_KO_s001
                                 + 0.606757 w_k1_MAT_s001
                                 + 0.938287 w_k1_MSFT_s001
                                 + 0.962462 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.780901 w_k1_ADM_s001
                                 + 0.778629 w_k1_AXP_s001
                                 + 0.705956 w_k1_DIS_s001
                                 + 0.750313 w_k1_FDX_s001
                                 + 0.915925 w_k1_IBM_s001
                                 + 0.975557 w_k1_INTC_s001
                                 + 0.773345 w_k1_KO_s001
                                 + 0.706142 w_k1_MAT_s001
                                 + 0.767138 w_k1_MSFT_s001
                                 + 0.658733 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.2639 w_k1_ADM_s001
                                 + 1.56442 w_k1_AXP_s001
                                 + 1.63365 w_k1_DIS_s001
                                 + 1.22152 w_k1_FDX_s001
                                 + 1.42011 w_k1_IBM_s001
                                 + 1.24203 w_k1_INTC_s001
                                 + 1.61684 w_k1_KO_s001 + 1.64848 w_k1_MAT_s001
                                 + 1.30688 w_k1_MSFT_s001
                                 + 1.17169 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.796471 w_k1_ADM_s002
                                 + 0.791568 w_k1_AXP_s002
                                 + 0.797315 w_k1_DIS_s002
                                 + 0.803574 w_k1_FDX_s002
                                 + 1.21256 w_k1_IBM_s002
                                 + 0.692057 w_k1_INTC_s002
                                 + 0.919806 w_k1_KO_s002
                                 + 1.36804 w_k1_MAT_s002
                                 + 0.764817 w_k1_MSFT_s002
                                 + 0.878486 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.18806 w_k1_ADM_s002
                                 + 1.45314 w_k1_AXP_s002
                                 + 1.58805 w_k1_DIS_s002 + 1.396 w_k1_FDX_s002
                                 + 0.987081 w_k1_IBM_s002
                                 + 1.07542 w_k1_INTC_s002
                                 + 1.08895 w_k1_KO_s002 + 1.47644 w_k1_MAT_s002
                                 + 1.26206 w_k1_MSFT_s002
                                 + 1.24427 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.05869 w_k1_ADM_s002
                                 + 0.812145 w_k1_AXP_s002
                                 + 0.807693 w_k1_DIS_s002
                                 + 0.900959 w_k1_FDX_s002
                                 + 0.664945 w_k1_IBM_s002
                                 + 0.823566 w_k1_INTC_s002
                                 + 0.765955 w_k1_KO_s002
                                 + 0.584389 w_k1_MAT_s002
                                 + 0.839874 w_k1_MSFT_s002
                                 + 0.877556 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.90791 w_k1_ADM_s002
                                 + 0.962571 w_k1_AXP_s002
                                 + 1.12727 w_k1_DIS_s002
                                 + 0.942054 w_k1_FDX_s002
                                 + 1.26357 w_k1_IBM_s002
                                 + 1.00123 w_k1_INTC_s002
                                 + 1.20914 w_k1_KO_s002 + 1.47512 w_k1_MAT_s002
                                 + 1.0301 w_k1_MSFT_s002 + 1.01001 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.638665 w_k1_ADM_s002
                                 + 0.595225 w_k1_AXP_s002
                                 + 0.667116 w_k1_DIS_s002
                                 + 0.710464 w_k1_FDX_s002
                                 + 0.779067 w_k1_IBM_s002
                                 + 0.880679 w_k1_INTC_s002
                                 + 0.879142 w_k1_KO_s002
                                 + 0.982952 w_k1_MAT_s002
                                 + 0.827386 w_k1_MSFT_s002
                                 + 0.932359 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.11482 w_k1_ADM_s002
                                 + 1.48948 w_k1_AXP_s002
                                 + 1.67558 w_k1_DIS_s002
                                 + 1.08544 w_k1_FDX_s002
                                 + 1.69116 w_k1_IBM_s002
                                 + 0.981931 w_k1_INTC_s002
                                 + 1.47535 w_k1_KO_s002 + 1.96885 w_k1_MAT_s002
                                 + 1.32287 w_k1_MSFT_s002 + 1.2817 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01766 w_k1_ADM_s002
                                 + 1.40487 w_k1_AXP_s002
                                 + 1.05921 w_k1_DIS_s002
                                 + 1.12535 w_k1_FDX_s002
                                 + 0.70935 w_k1_IBM_s002
                                 + 0.996535 w_k1_INTC_s002
                                 + 0.910749 w_k1_KO_s002
                                 + 0.921265 w_k1_MAT_s002
                                 + 1.04714 w_k1_MSFT_s002
                                 + 0.920042 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.07926 w_k1_ADM_s002
                                 + 1.11369 w_k1_AXP_s002
                                 + 0.984697 w_k1_DIS_s002
                                 + 1.08775 w_k1_FDX_s002
                                 + 0.875875 w_k1_IBM_s002
                                 + 0.982853 w_k1_INTC_s002
                                 + 0.969853 w_k1_KO_s002
                                 + 1.04012 w_k1_MAT_s002
                                 + 0.894724 w_k1_MSFT_s002
                                 + 0.990307 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.817155 w_k1_ADM_s003
                                 + 0.67288 w_k1_AXP_s003
                                 + 0.83565 w_k1_DIS_s003
                                 + 0.775253 w_k1_FDX_s003
                                 + 1.14086 w_k1_IBM_s003
                                 + 0.752014 w_k1_INTC_s003
                                 + 1.02346 w_k1_KO_s003 + 1.66279 w_k1_MAT_s003
                                 + 0.879677 w_k1_MSFT_s003
                                 + 0.927714 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.835205 w_k1_ADM_s003
                                 + 0.83772 w_k1_AXP_s003
                                 + 0.958545 w_k1_DIS_s003
                                 + 0.870624 w_k1_FDX_s003
                                 + 1.64374 w_k1_IBM_s003
                                 + 0.987324 w_k1_INTC_s003
                                 + 1.0467 w_k1_KO_s003 + 1.68783 w_k1_MAT_s003
                                 + 1.043 w_k1_MSFT_s003 + 1.14294 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.12781 w_k1_ADM_s003
                                 + 1.19937 w_k1_AXP_s003
                                 + 1.34316 w_k1_DIS_s003
                                 + 1.29376 w_k1_FDX_s003
                                 + 1.06055 w_k1_IBM_s003
                                 + 1.10915 w_k1_INTC_s003
                                 + 1.01686 w_k1_KO_s003
                                 + 0.973078 w_k1_MAT_s003
                                 + 1.2011 w_k1_MSFT_s003 + 1.15383 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.968422 w_k1_ADM_s003
                                 + 0.998386 w_k1_AXP_s003
                                 + 0.802477 w_k1_DIS_s003
                                 + 0.978211 w_k1_FDX_s003
                                 + 0.952029 w_k1_IBM_s003
                                 + 0.878956 w_k1_INTC_s003
                                 + 0.80757 w_k1_KO_s003 + 1.34947 w_k1_MAT_s003
                                 + 0.930005 w_k1_MSFT_s003
                                 + 0.970141 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.87789 w_k1_ADM_s003
                                 + 0.799737 w_k1_AXP_s003
                                 + 0.912041 w_k1_DIS_s003
                                 + 0.978288 w_k1_FDX_s003
                                 + 0.759157 w_k1_IBM_s003
                                 + 0.818179 w_k1_INTC_s003
                                 + 0.743554 w_k1_KO_s003
                                 + 0.842018 w_k1_MAT_s003
                                 + 0.911326 w_k1_MSFT_s003
                                 + 0.871864 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.987876 w_k1_ADM_s003
                                 + 1.56249 w_k1_AXP_s003
                                 + 1.05046 w_k1_DIS_s003
                                 + 1.14866 w_k1_FDX_s003
                                 + 1.34644 w_k1_IBM_s003
                                 + 0.924767 w_k1_INTC_s003
                                 + 1.12551 w_k1_KO_s003 + 2.2144 w_k1_MAT_s003
                                 + 0.806868 w_k1_MSFT_s003
                                 + 1.08916 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.13299 w_k1_ADM_s003
                                 + 1.33146 w_k1_AXP_s003
                                 + 1.33261 w_k1_DIS_s003
                                 + 1.34047 w_k1_FDX_s003
                                 + 0.701429 w_k1_IBM_s003
                                 + 0.966789 w_k1_INTC_s003
                                 + 0.996311 w_k1_KO_s003
                                 + 0.996789 w_k1_MAT_s003
                                 + 1.19898 w_k1_MSFT_s003
                                 + 1.06136 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.821374 w_k1_ADM_s003
                                 + 0.808693 w_k1_AXP_s003
                                 + 0.571586 w_k1_DIS_s003
                                 + 0.762586 w_k1_FDX_s003
                                 + 0.789867 w_k1_IBM_s003
                                 + 0.717081 w_k1_INTC_s003
                                 + 0.803193 w_k1_KO_s003
                                 + 1.22161 w_k1_MAT_s003
                                 + 0.799243 w_k1_MSFT_s003
                                 + 0.858929 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.13274 w_k1_ADM_s004
                                 + 1.25087 w_k1_AXP_s004
                                 + 1.25654 w_k1_DIS_s004
                                 + 1.25165 w_k1_FDX_s004
                                 + 0.684274 w_k1_IBM_s004
                                 + 1.33526 w_k1_INTC_s004
                                 + 0.922192 w_k1_KO_s004
                                 + 0.573189 w_k1_MAT_s004
                                 + 1.33532 w_k1_MSFT_s004
                                 + 0.996343 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.641162 w_k1_ADM_s004
                                 + 1.20697 w_k1_AXP_s004
                                 + 0.874838 w_k1_DIS_s004
                                 + 0.942304 w_k1_FDX_s004
                                 + 1.24751 w_k1_IBM_s004
                                 + 1.04621 w_k1_INTC_s004 + 0.8723 w_k1_KO_s004
                                 + 1.15638 w_k1_MAT_s004
                                 + 0.839042 w_k1_MSFT_s004
                                 + 0.978222 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.12007 w_k1_ADM_s004
                                 + 1.47656 w_k1_AXP_s004
                                 + 1.31887 w_k1_DIS_s004
                                 + 1.19997 w_k1_FDX_s004
                                 + 1.16406 w_k1_IBM_s004
                                 + 1.29959 w_k1_INTC_s004
                                 + 1.11743 w_k1_KO_s004 + 1.44255 w_k1_MAT_s004
                                 + 1.19264 w_k1_MSFT_s004 + 1.1246 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.874932 w_k1_ADM_s004
                                 + 1.02388 w_k1_AXP_s004
                                 + 1.29859 w_k1_DIS_s004
                                 + 0.979011 w_k1_FDX_s004
                                 + 1.13938 w_k1_IBM_s004
                                 + 1.21147 w_k1_INTC_s004
                                 + 1.19902 w_k1_KO_s004 + 1.10519 w_k1_MAT_s004
                                 + 1.10223 w_k1_MSFT_s004
                                 + 1.05414 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.68666 w_k1_ADM_s004
                                 + 0.722325 w_k1_AXP_s004
                                 + 0.668417 w_k1_DIS_s004
                                 + 0.685056 w_k1_FDX_s004
                                 + 1.12742 w_k1_IBM_s004
                                 + 0.897566 w_k1_INTC_s004
                                 + 0.736428 w_k1_KO_s004
                                 + 1.18174 w_k1_MAT_s004
                                 + 0.774931 w_k1_MSFT_s004
                                 + 0.817765 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.06492 w_k1_ADM_s004
                                 + 0.954405 w_k1_AXP_s004
                                 + 0.850674 w_k1_DIS_s004
                                 + 1.0385 w_k1_FDX_s004
                                 + 0.711626 w_k1_IBM_s004
                                 + 0.907658 w_k1_INTC_s004
                                 + 0.818953 w_k1_KO_s004
                                 + 0.782529 w_k1_MAT_s004
                                 + 0.984238 w_k1_MSFT_s004
                                 + 0.995311 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.729435 w_k1_ADM_s004
                                 + 0.795751 w_k1_AXP_s004
                                 + 0.68015 w_k1_DIS_s004
                                 + 0.733936 w_k1_FDX_s004
                                 + 0.751509 w_k1_IBM_s004
                                 + 0.916682 w_k1_INTC_s004
                                 + 0.728142 w_k1_KO_s004
                                 + 0.833696 w_k1_MAT_s004
                                 + 0.814505 w_k1_MSFT_s004
                                 + 0.80546 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.765191 w_k1_ADM_s004
                                 + 0.981627 w_k1_AXP_s004
                                 + 1.03117 w_k1_DIS_s004
                                 + 0.797381 w_k1_FDX_s004
                                 + 1.84682 w_k1_IBM_s004
                                 + 0.909793 w_k1_INTC_s004
                                 + 1.21121 w_k1_KO_s004 + 1.42797 w_k1_MAT_s004
                                 + 0.810613 w_k1_MSFT_s004
                                 + 0.887148 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.51596 w_k1_ADM_s005
                                 + 1.37665 w_k1_AXP_s005
                                 + 1.44213 w_k1_DIS_s005
                                 + 1.28682 w_k1_FDX_s005
                                 + 0.924973 w_k1_IBM_s005
                                 + 1.2252 w_k1_INTC_s005 + 1.36456 w_k1_KO_s005
                                 + 0.899014 w_k1_MAT_s005
                                 + 1.38012 w_k1_MSFT_s005
                                 + 1.02825 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.12071 w_k1_ADM_s005
                                 + 0.852695 w_k1_AXP_s005
                                 + 0.818779 w_k1_DIS_s005
                                 + 0.920098 w_k1_FDX_s005
                                 + 0.739903 w_k1_IBM_s005
                                 + 1.06176 w_k1_INTC_s005
                                 + 0.956815 w_k1_KO_s005
                                 + 0.492453 w_k1_MAT_s005
                                 + 0.996373 w_k1_MSFT_s005
                                 + 0.861849 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.997556 w_k1_ADM_s005
                                 + 0.90231 w_k1_AXP_s005
                                 + 0.875721 w_k1_DIS_s005
                                 + 0.824588 w_k1_FDX_s005
                                 + 1.58088 w_k1_IBM_s005
                                 + 0.764066 w_k1_INTC_s005
                                 + 1.15418 w_k1_KO_s005 + 1.48326 w_k1_MAT_s005
                                 + 0.874386 w_k1_MSFT_s005
                                 + 0.888307 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.31402 w_k1_ADM_s005
                                 + 1.27655 w_k1_AXP_s005
                                 + 1.32145 w_k1_DIS_s005
                                 + 1.18578 w_k1_FDX_s005
                                 + 1.66369 w_k1_IBM_s005
                                 + 0.908982 w_k1_INTC_s005
                                 + 1.67401 w_k1_KO_s005 + 1.8697 w_k1_MAT_s005
                                 + 1.07571 w_k1_MSFT_s005
                                 + 1.21625 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.654591 w_k1_ADM_s005
                                 + 0.703984 w_k1_AXP_s005
                                 + 0.652682 w_k1_DIS_s005
                                 + 0.646698 w_k1_FDX_s005
                                 + 0.981327 w_k1_IBM_s005
                                 + 0.817205 w_k1_INTC_s005
                                 + 0.922391 w_k1_KO_s005
                                 + 0.847884 w_k1_MAT_s005
                                 + 0.693885 w_k1_MSFT_s005
                                 + 0.786633 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.08352 w_k1_ADM_s005
                                 + 1.34692 w_k1_AXP_s005
                                 + 1.54307 w_k1_DIS_s005
                                 + 1.51722 w_k1_FDX_s005
                                 + 0.622463 w_k1_IBM_s005
                                 + 1.6031 w_k1_INTC_s005 + 1.23792 w_k1_KO_s005
                                 + 0.934734 w_k1_MAT_s005
                                 + 1.51011 w_k1_MSFT_s005
                                 + 1.03837 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.26036 w_k1_ADM_s005
                                 + 1.36717 w_k1_AXP_s005
                                 + 1.25269 w_k1_DIS_s005
                                 + 1.13166 w_k1_FDX_s005
                                 + 1.08755 w_k1_IBM_s005
                                 + 0.87854 w_k1_INTC_s005
                                 + 1.28074 w_k1_KO_s005 + 1.15353 w_k1_MAT_s005
                                 + 1.15684 w_k1_MSFT_s005
                                 + 0.99898 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.649869 w_k1_ADM_s005
                                 + 0.961552 w_k1_AXP_s005
                                 + 0.86952 w_k1_DIS_s005
                                 + 0.919938 w_k1_FDX_s005
                                 + 1.1261 w_k1_IBM_s005
                                 + 1.02834 w_k1_INTC_s005
                                 + 1.13949 w_k1_KO_s005 + 1.14992 w_k1_MAT_s005
                                 + 0.897829 w_k1_MSFT_s005
                                 + 0.925587 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.34121 w_k1_ADM_s006
                                 + 1.03394 w_k1_AXP_s006
                                 + 1.03361 w_k1_DIS_s006
                                 + 1.26844 w_k1_FDX_s006
                                 + 0.763386 w_k1_IBM_s006
                                 + 1.14403 w_k1_INTC_s006
                                 + 1.08147 w_k1_KO_s006
                                 + 0.940156 w_k1_MAT_s006
                                 + 1.21471 w_k1_MSFT_s006
                                 + 1.12422 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.18345 w_k1_ADM_s006
                                 + 0.950004 w_k1_AXP_s006
                                 + 0.992507 w_k1_DIS_s006
                                 + 0.891707 w_k1_FDX_s006
                                 + 1.4378 w_k1_IBM_s006
                                 + 1.07228 w_k1_INTC_s006
                                 + 1.17931 w_k1_KO_s006 + 1.30842 w_k1_MAT_s006
                                 + 1.01933 w_k1_MSFT_s006
                                 + 1.06299 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.624 w_k1_ADM_s006
                                 + 1.62154 w_k1_AXP_s006
                                 + 1.35269 w_k1_DIS_s006
                                 + 1.15629 w_k1_FDX_s006
                                 + 1.68155 w_k1_IBM_s006
                                 + 1.31523 w_k1_INTC_s006
                                 + 1.59301 w_k1_KO_s006 + 1.87506 w_k1_MAT_s006
                                 + 1.1395 w_k1_MSFT_s006 + 1.26455 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.07215 w_k1_ADM_s006
                                 + 0.61627 w_k1_AXP_s006
                                 + 0.812703 w_k1_DIS_s006
                                 + 0.997555 w_k1_FDX_s006
                                 + 0.545574 w_k1_IBM_s006
                                 + 1.17836 w_k1_INTC_s006
                                 + 0.740218 w_k1_KO_s006
                                 + 0.477751 w_k1_MAT_s006
                                 + 0.939065 w_k1_MSFT_s006
                                 + 0.810163 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.94532 w_k1_ADM_s006
                                 + 0.83225 w_k1_AXP_s006
                                 + 0.834777 w_k1_DIS_s006
                                 + 0.894406 w_k1_FDX_s006
                                 + 0.854849 w_k1_IBM_s006
                                 + 1.06687 w_k1_INTC_s006
                                 + 0.875869 w_k1_KO_s006
                                 + 0.714869 w_k1_MAT_s006
                                 + 0.901544 w_k1_MSFT_s006
                                 + 0.952144 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.35075 w_k1_ADM_s006
                                 + 1.25964 w_k1_AXP_s006
                                 + 1.47493 w_k1_DIS_s006
                                 + 1.18512 w_k1_FDX_s006 + 1.2829 w_k1_IBM_s006
                                 + 1.12919 w_k1_INTC_s006
                                 + 1.51606 w_k1_KO_s006 + 1.31946 w_k1_MAT_s006
                                 + 1.14833 w_k1_MSFT_s006
                                 + 1.26905 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.15114 w_k1_ADM_s006
                                 + 1.02577 w_k1_AXP_s006
                                 + 1.05467 w_k1_DIS_s006 + 1.0006 w_k1_FDX_s006
                                 + 1.06928 w_k1_IBM_s006
                                 + 1.09283 w_k1_INTC_s006
                                 + 1.16675 w_k1_KO_s006 + 1.35863 w_k1_MAT_s006
                                 + 1.05865 w_k1_MSFT_s006
                                 + 1.13721 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.949278 w_k1_ADM_s006
                                 + 0.648741 w_k1_AXP_s006
                                 + 0.57359 w_k1_DIS_s006
                                 + 0.769136 w_k1_FDX_s006
                                 + 0.597145 w_k1_IBM_s006
                                 + 0.839628 w_k1_INTC_s006
                                 + 0.708025 w_k1_KO_s006
                                 + 0.636501 w_k1_MAT_s006
                                 + 0.823832 w_k1_MSFT_s006
                                 + 0.911277 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.06247 w_k1_ADM_s007
                                 + 0.811679 w_k1_AXP_s007
                                 + 0.776925 w_k1_DIS_s007
                                 + 0.796853 w_k1_FDX_s007
                                 + 0.818807 w_k1_IBM_s007
                                 + 0.797959 w_k1_INTC_s007
                                 + 0.928665 w_k1_KO_s007
                                 + 0.733407 w_k1_MAT_s007
                                 + 0.76551 w_k1_MSFT_s007
                                 + 0.776644 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.948698 w_k1_ADM_s007
                                 + 1.12391 w_k1_AXP_s007
                                 + 1.23892 w_k1_DIS_s007
                                 + 1.14252 w_k1_FDX_s007
                                 + 1.07038 w_k1_IBM_s007
                                 + 1.34271 w_k1_INTC_s007
                                 + 1.07225 w_k1_KO_s007 + 1.0932 w_k1_MAT_s007
                                 + 1.16349 w_k1_MSFT_s007
                                 + 0.923612 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.962783 w_k1_ADM_s007
                                 + 0.805894 w_k1_AXP_s007
                                 + 0.938604 w_k1_DIS_s007
                                 + 0.98411 w_k1_FDX_s007
                                 + 0.540998 w_k1_IBM_s007
                                 + 0.874059 w_k1_INTC_s007
                                 + 0.809606 w_k1_KO_s007
                                 + 0.668143 w_k1_MAT_s007
                                 + 0.929654 w_k1_MSFT_s007
                                 + 0.821354 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.43605 w_k1_ADM_s007
                                 + 1.46082 w_k1_AXP_s007
                                 + 1.87199 w_k1_DIS_s007
                                 + 1.45092 w_k1_FDX_s007
                                 + 1.14702 w_k1_IBM_s007
                                 + 1.32385 w_k1_INTC_s007
                                 + 1.42237 w_k1_KO_s007 + 1.11609 w_k1_MAT_s007
                                 + 1.32193 w_k1_MSFT_s007
                                 + 1.20096 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.2025 w_k1_ADM_s007
                                 + 1.28889 w_k1_AXP_s007
                                 + 1.36671 w_k1_DIS_s007
                                 + 1.27784 w_k1_FDX_s007
                                 + 0.775696 w_k1_IBM_s007
                                 + 1.18308 w_k1_INTC_s007
                                 + 0.892644 w_k1_KO_s007
                                 + 0.71548 w_k1_MAT_s007
                                 + 1.2789 w_k1_MSFT_s007 + 1.07841 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.04886 w_k1_ADM_s007
                                 + 1.25322 w_k1_AXP_s007
                                 + 1.31616 w_k1_DIS_s007
                                 + 1.01613 w_k1_FDX_s007 + 1.725 w_k1_IBM_s007
                                 + 1.25704 w_k1_INTC_s007
                                 + 1.54316 w_k1_KO_s007 + 1.56731 w_k1_MAT_s007
                                 + 1.07623 w_k1_MSFT_s007
                                 + 1.08787 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.730106 w_k1_ADM_s007
                                 + 1.0476 w_k1_AXP_s007
                                 + 0.858075 w_k1_DIS_s007
                                 + 0.782879 w_k1_FDX_s007
                                 + 1.28066 w_k1_IBM_s007
                                 + 0.875192 w_k1_INTC_s007
                                 + 1.10813 w_k1_KO_s007 + 1.47327 w_k1_MAT_s007
                                 + 0.83216 w_k1_MSFT_s007
                                 + 0.911591 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.956907 w_k1_ADM_s007
                                 + 1.03367 w_k1_AXP_s007 + 1.0072 w_k1_DIS_s007
                                 + 0.937939 w_k1_FDX_s007
                                 + 0.858757 w_k1_IBM_s007
                                 + 0.994406 w_k1_INTC_s007
                                 + 1.05357 w_k1_KO_s007 + 1.12947 w_k1_MAT_s007
                                 + 0.892263 w_k1_MSFT_s007
                                 + 0.882291 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.984514 w_k1_ADM_s010
                                 + 1.13992 w_k1_AXP_s010
                                 + 0.865444 w_k1_DIS_s010
                                 + 1.02641 w_k1_FDX_s010
                                 + 0.983234 w_k1_IBM_s010
                                 + 1.05927 w_k1_INTC_s010
                                 + 0.831806 w_k1_KO_s010
                                 + 0.899257 w_k1_MAT_s010
                                 + 0.8745 w_k1_MSFT_s010
                                 + 0.952997 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.19907 w_k1_ADM_s010
                                 + 1.3382 w_k1_AXP_s010 + 1.51564 w_k1_DIS_s010
                                 + 1.18919 w_k1_FDX_s010 + 1.3875 w_k1_IBM_s010
                                 + 1.15926 w_k1_INTC_s010
                                 + 1.35988 w_k1_KO_s010 + 1.60609 w_k1_MAT_s010
                                 + 1.08756 w_k1_MSFT_s010
                                 + 1.29034 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.91155 w_k1_ADM_s010
                                 + 0.645459 w_k1_AXP_s010
                                 + 0.813428 w_k1_DIS_s010
                                 + 0.830829 w_k1_FDX_s010
                                 + 0.701645 w_k1_IBM_s010
                                 + 1.12206 w_k1_INTC_s010
                                 + 0.687109 w_k1_KO_s010
                                 + 0.381786 w_k1_MAT_s010
                                 + 0.944711 w_k1_MSFT_s010
                                 + 0.889395 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.12666 w_k1_ADM_s010
                                 + 0.957235 w_k1_AXP_s010
                                 + 1.17082 w_k1_DIS_s010 + 1.0117 w_k1_FDX_s010
                                 + 1.0831 w_k1_IBM_s010
                                 + 1.32612 w_k1_INTC_s010
                                 + 1.01057 w_k1_KO_s010
                                 + 0.620227 w_k1_MAT_s010
                                 + 0.981693 w_k1_MSFT_s010
                                 + 0.98106 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.91345 w_k1_ADM_s010
                                 + 0.901019 w_k1_AXP_s010
                                 + 0.590667 w_k1_DIS_s010
                                 + 0.734219 w_k1_FDX_s010
                                 + 0.869907 w_k1_IBM_s010
                                 + 0.872498 w_k1_INTC_s010
                                 + 0.817263 w_k1_KO_s010
                                 + 0.78121 w_k1_MAT_s010
                                 + 0.701435 w_k1_MSFT_s010
                                 + 0.852527 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.03849 w_k1_ADM_s010
                                 + 0.924554 w_k1_AXP_s010
                                 + 0.868884 w_k1_DIS_s010
                                 + 0.848249 w_k1_FDX_s010
                                 + 1.03903 w_k1_IBM_s010
                                 + 0.91563 w_k1_INTC_s010
                                 + 0.966352 w_k1_KO_s010
                                 + 1.10413 w_k1_MAT_s010
                                 + 0.895823 w_k1_MSFT_s010
                                 + 1.00424 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.813031 w_k1_ADM_s010
                                 + 0.552639 w_k1_AXP_s010
                                 + 0.610347 w_k1_DIS_s010
                                 + 0.650959 w_k1_FDX_s010
                                 + 0.736073 w_k1_IBM_s010
                                 + 0.787237 w_k1_INTC_s010
                                 + 0.777318 w_k1_KO_s010
                                 + 0.599351 w_k1_MAT_s010
                                 + 0.687756 w_k1_MSFT_s010
                                 + 0.834334 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.35994 w_k1_ADM_s010
                                 + 1.34017 w_k1_AXP_s010
                                 + 1.32156 w_k1_DIS_s010 + 1.2985 w_k1_FDX_s010
                                 + 0.886285 w_k1_IBM_s010
                                 + 1.38395 w_k1_INTC_s010
                                 + 1.12471 w_k1_KO_s010
                                 + 0.699554 w_k1_MAT_s010
                                 + 1.04417 w_k1_MSFT_s010
                                 + 1.12356 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.849798 w_k1_ADM_s011
                                 + 0.848001 w_k1_AXP_s011
                                 + 0.767037 w_k1_DIS_s011
                                 + 0.775966 w_k1_FDX_s011
                                 + 1.23045 w_k1_IBM_s011
                                 + 0.984811 w_k1_INTC_s011
                                 + 0.932361 w_k1_KO_s011 + 1.0823 w_k1_MAT_s011
                                 + 0.849706 w_k1_MSFT_s011
                                 + 0.847121 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.09286 w_k1_ADM_s011
                                 + 1.22275 w_k1_AXP_s011 + 1.4972 w_k1_DIS_s011
                                 + 1.21997 w_k1_FDX_s011
                                 + 0.951376 w_k1_IBM_s011
                                 + 1.29706 w_k1_INTC_s011
                                 + 1.15009 w_k1_KO_s011 + 1.28716 w_k1_MAT_s011
                                 + 1.16759 w_k1_MSFT_s011
                                 + 1.11013 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.936385 w_k1_ADM_s011
                                 + 0.690678 w_k1_AXP_s011
                                 + 0.896825 w_k1_DIS_s011
                                 + 0.875164 w_k1_FDX_s011
                                 + 0.683701 w_k1_IBM_s011
                                 + 0.944529 w_k1_INTC_s011
                                 + 0.789787 w_k1_KO_s011
                                 + 0.612329 w_k1_MAT_s011
                                 + 0.90102 w_k1_MSFT_s011
                                 + 0.819032 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.00205 w_k1_ADM_s011
                                 + 1.17224 w_k1_AXP_s011
                                 + 0.805522 w_k1_DIS_s011
                                 + 1.03525 w_k1_FDX_s011
                                 + 0.882601 w_k1_IBM_s011
                                 + 1.08935 w_k1_INTC_s011
                                 + 0.945934 w_k1_KO_s011
                                 + 1.12863 w_k1_MAT_s011
                                 + 0.923521 w_k1_MSFT_s011
                                 + 0.854755 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.41306 w_k1_ADM_s011
                                 + 1.47305 w_k1_AXP_s011
                                 + 1.34819 w_k1_DIS_s011 + 1.3164 w_k1_FDX_s011
                                 + 1.02419 w_k1_IBM_s011
                                 + 1.27205 w_k1_INTC_s011
                                 + 1.07837 w_k1_KO_s011 + 1.04373 w_k1_MAT_s011
                                 + 1.10465 w_k1_MSFT_s011
                                 + 1.07515 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.998928 w_k1_ADM_s011
                                 + 0.918579 w_k1_AXP_s011
                                 + 1.00204 w_k1_DIS_s011
                                 + 0.960824 w_k1_FDX_s011
                                 + 1.33119 w_k1_IBM_s011
                                 + 0.972031 w_k1_INTC_s011
                                 + 1.06908 w_k1_KO_s011 + 1.34809 w_k1_MAT_s011
                                 + 0.946745 w_k1_MSFT_s011
                                 + 0.851888 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.853476 w_k1_ADM_s011
                                 + 0.62744 w_k1_AXP_s011
                                 + 0.591932 w_k1_DIS_s011
                                 + 0.692861 w_k1_FDX_s011
                                 + 0.778834 w_k1_IBM_s011
                                 + 0.683156 w_k1_INTC_s011
                                 + 0.63641 w_k1_KO_s011 + 0.65978 w_k1_MAT_s011
                                 + 0.679478 w_k1_MSFT_s011
                                 + 0.701378 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.02385 w_k1_ADM_s011
                                 + 0.739531 w_k1_AXP_s011
                                 + 1.09037 w_k1_DIS_s011
                                 + 1.07645 w_k1_FDX_s011
                                 + 0.630754 w_k1_IBM_s011
                                 + 1.05466 w_k1_INTC_s011
                                 + 0.934473 w_k1_KO_s011
                                 + 0.630131 w_k1_MAT_s011
                                 + 1.10302 w_k1_MSFT_s011
                                 + 0.898668 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.25421 w_k1_ADM_s012
                                 + 1.14978 w_k1_AXP_s012
                                 + 1.06515 w_k1_DIS_s012
                                 + 1.22957 w_k1_FDX_s012
                                 + 0.818176 w_k1_IBM_s012
                                 + 1.15788 w_k1_INTC_s012
                                 + 1.02336 w_k1_KO_s012
                                 + 0.801621 w_k1_MAT_s012
                                 + 0.998878 w_k1_MSFT_s012
                                 + 1.07923 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.04048 w_k1_ADM_s012
                                 + 0.675443 w_k1_AXP_s012
                                 + 0.848498 w_k1_DIS_s012
                                 + 0.87648 w_k1_FDX_s012
                                 + 0.701908 w_k1_IBM_s012
                                 + 0.945789 w_k1_INTC_s012
                                 + 0.820311 w_k1_KO_s012
                                 + 0.540281 w_k1_MAT_s012
                                 + 0.883864 w_k1_MSFT_s012
                                 + 0.875027 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.28939 w_k1_ADM_s012
                                 + 1.23983 w_k1_AXP_s012
                                 + 1.42899 w_k1_DIS_s012
                                 + 1.26289 w_k1_FDX_s012 + 1.3317 w_k1_IBM_s012
                                 + 1.03885 w_k1_INTC_s012
                                 + 1.31018 w_k1_KO_s012 + 1.43149 w_k1_MAT_s012
                                 + 1.13453 w_k1_MSFT_s012 + 1.3011 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.95774 w_k1_ADM_s012
                                 + 0.852148 w_k1_AXP_s012
                                 + 0.926306 w_k1_DIS_s012
                                 + 0.931975 w_k1_FDX_s012
                                 + 0.558816 w_k1_IBM_s012
                                 + 0.782734 w_k1_INTC_s012
                                 + 0.715434 w_k1_KO_s012
                                 + 0.477397 w_k1_MAT_s012
                                 + 0.826749 w_k1_MSFT_s012
                                 + 0.841796 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.09025 w_k1_ADM_s012
                                 + 1.11113 w_k1_AXP_s012
                                 + 1.50954 w_k1_DIS_s012
                                 + 1.28085 w_k1_FDX_s012
                                 + 1.03749 w_k1_IBM_s012
                                 + 1.28847 w_k1_INTC_s012
                                 + 1.16689 w_k1_KO_s012 + 1.13308 w_k1_MAT_s012
                                 + 1.28169 w_k1_MSFT_s012
                                 + 1.12332 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.995837 w_k1_ADM_s012
                                 + 0.748531 w_k1_AXP_s012
                                 + 0.680784 w_k1_DIS_s012
                                 + 0.77366 w_k1_FDX_s012
                                 + 0.88569 w_k1_IBM_s012
                                 + 0.762025 w_k1_INTC_s012
                                 + 0.82772 w_k1_KO_s012
                                 + 0.891563 w_k1_MAT_s012
                                 + 0.770155 w_k1_MSFT_s012
                                 + 0.853712 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.924907 w_k1_ADM_s012
                                 + 0.966198 w_k1_AXP_s012
                                 + 0.822964 w_k1_DIS_s012
                                 + 0.863349 w_k1_FDX_s012
                                 + 1.19344 w_k1_IBM_s012
                                 + 1.27308 w_k1_INTC_s012
                                 + 1.18058 w_k1_KO_s012 + 1.06443 w_k1_MAT_s012
                                 + 0.938971 w_k1_MSFT_s012
                                 + 0.931207 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.81788 w_k1_ADM_s012
                                 + 0.916341 w_k1_AXP_s012
                                 + 0.734896 w_k1_DIS_s012
                                 + 0.86204 w_k1_FDX_s012
                                 + 0.724046 w_k1_IBM_s012
                                 + 0.924738 w_k1_INTC_s012
                                 + 0.862449 w_k1_KO_s012
                                 + 0.706212 w_k1_MAT_s012
                                 + 0.729605 w_k1_MSFT_s012
                                 + 0.748101 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.999069 w_k1_ADM_s013
                                 + 0.842198 w_k1_AXP_s013
                                 + 0.877783 w_k1_DIS_s013
                                 + 0.933431 w_k1_FDX_s013
                                 + 0.789528 w_k1_IBM_s013
                                 + 1.07629 w_k1_INTC_s013
                                 + 0.92563 w_k1_KO_s013
                                 + 0.785823 w_k1_MAT_s013
                                 + 0.99927 w_k1_MSFT_s013
                                 + 0.847359 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.43096 w_k1_ADM_s013
                                 + 1.41153 w_k1_AXP_s013
                                 + 1.76935 w_k1_DIS_s013
                                 + 1.45572 w_k1_FDX_s013
                                 + 1.04426 w_k1_IBM_s013
                                 + 1.33452 w_k1_INTC_s013
                                 + 1.34654 w_k1_KO_s013
                                 + 0.965175 w_k1_MAT_s013
                                 + 1.46837 w_k1_MSFT_s013
                                 + 1.22732 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.03856 w_k1_ADM_s013
                                 + 0.991296 w_k1_AXP_s013
                                 + 1.24536 w_k1_DIS_s013
                                 + 1.14927 w_k1_FDX_s013
                                 + 1.12736 w_k1_IBM_s013
                                 + 1.20083 w_k1_INTC_s013
                                 + 1.27775 w_k1_KO_s013
                                 + 0.966191 w_k1_MAT_s013
                                 + 1.18663 w_k1_MSFT_s013
                                 + 1.07328 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.829889 w_k1_ADM_s013
                                 + 0.823001 w_k1_AXP_s013
                                 + 0.6651 w_k1_DIS_s013
                                 + 0.771635 w_k1_FDX_s013
                                 + 0.94973 w_k1_IBM_s013
                                 + 1.02536 w_k1_INTC_s013
                                 + 0.816631 w_k1_KO_s013
                                 + 0.85792 w_k1_MAT_s013
                                 + 0.786879 w_k1_MSFT_s013
                                 + 0.745569 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.03306 w_k1_ADM_s013
                                 + 0.95245 w_k1_AXP_s013
                                 + 0.729754 w_k1_DIS_s013
                                 + 0.99678 w_k1_FDX_s013
                                 + 0.966644 w_k1_IBM_s013
                                 + 1.00534 w_k1_INTC_s013
                                 + 0.935169 w_k1_KO_s013
                                 + 0.989945 w_k1_MAT_s013
                                 + 0.975777 w_k1_MSFT_s013
                                 + 0.865146 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.19232 w_k1_ADM_s013
                                 + 1.47518 w_k1_AXP_s013
                                 + 1.09364 w_k1_DIS_s013
                                 + 1.08121 w_k1_FDX_s013
                                 + 1.28116 w_k1_IBM_s013
                                 + 1.22287 w_k1_INTC_s013
                                 + 1.12099 w_k1_KO_s013 + 1.26363 w_k1_MAT_s013
                                 + 1.01595 w_k1_MSFT_s013
                                 + 1.07284 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.941933 w_k1_ADM_s013
                                 + 1.05006 w_k1_AXP_s013
                                 + 1.07531 w_k1_DIS_s013
                                 + 0.988155 w_k1_FDX_s013
                                 + 1.02006 w_k1_IBM_s013
                                 + 1.22351 w_k1_INTC_s013
                                 + 1.23599 w_k1_KO_s013
                                 + 0.844459 w_k1_MAT_s013
                                 + 0.92124 w_k1_MSFT_s013
                                 + 0.979876 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.23614 w_k1_ADM_s013
                                 + 1.08065 w_k1_AXP_s013
                                 + 0.952111 w_k1_DIS_s013
                                 + 1.22341 w_k1_FDX_s013
                                 + 0.767256 w_k1_IBM_s013
                                 + 1.2868 w_k1_INTC_s013
                                 + 0.856219 w_k1_KO_s013
                                 + 0.780244 w_k1_MAT_s013
                                 + 1.04688 w_k1_MSFT_s013
                                 + 0.951548 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.3069 w_k1_ADM_s014
                                 + 1.05255 w_k1_AXP_s014
                                 + 0.841893 w_k1_DIS_s014
                                 + 1.00714 w_k1_FDX_s014
                                 + 0.751783 w_k1_IBM_s014
                                 + 0.983787 w_k1_INTC_s014
                                 + 0.719102 w_k1_KO_s014
                                 + 0.636278 w_k1_MAT_s014
                                 + 0.874259 w_k1_MSFT_s014
                                 + 0.920001 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.938279 w_k1_ADM_s014
                                 + 0.861006 w_k1_AXP_s014
                                 + 0.636517 w_k1_DIS_s014
                                 + 0.714063 w_k1_FDX_s014
                                 + 1.10624 w_k1_IBM_s014
                                 + 0.740645 w_k1_INTC_s014
                                 + 0.752475 w_k1_KO_s014
                                 + 1.16668 w_k1_MAT_s014
                                 + 0.720012 w_k1_MSFT_s014
                                 + 0.99578 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.57982 w_k1_ADM_s014
                                 + 1.28144 w_k1_AXP_s014
                                 + 1.48306 w_k1_DIS_s014
                                 + 1.30314 w_k1_FDX_s014
                                 + 1.61456 w_k1_IBM_s014
                                 + 1.06573 w_k1_INTC_s014
                                 + 1.32737 w_k1_KO_s014 + 1.51498 w_k1_MAT_s014
                                 + 1.14516 w_k1_MSFT_s014 + 1.1571 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.79943 w_k1_ADM_s014
                                 + 1.75162 w_k1_AXP_s014
                                 + 1.94887 w_k1_DIS_s014
                                 + 1.47029 w_k1_FDX_s014
                                 + 1.39776 w_k1_IBM_s014
                                 + 1.29144 w_k1_INTC_s014
                                 + 1.47268 w_k1_KO_s014 + 1.77029 w_k1_MAT_s014
                                 + 1.40838 w_k1_MSFT_s014
                                 + 1.37558 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.33872 w_k1_ADM_s014
                                 + 1.18092 w_k1_AXP_s014
                                 + 1.43919 w_k1_DIS_s014
                                 + 1.25292 w_k1_FDX_s014
                                 + 0.760456 w_k1_IBM_s014
                                 + 1.20896 w_k1_INTC_s014
                                 + 0.938031 w_k1_KO_s014
                                 + 0.770225 w_k1_MAT_s014
                                 + 1.17508 w_k1_MSFT_s014
                                 + 1.03329 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.15632 w_k1_ADM_s014
                                 + 1.32315 w_k1_AXP_s014
                                 + 1.11795 w_k1_DIS_s014
                                 + 1.04233 w_k1_FDX_s014
                                 + 1.30068 w_k1_IBM_s014
                                 + 1.14748 w_k1_INTC_s014 + 1.0977 w_k1_KO_s014
                                 + 1.08077 w_k1_MAT_s014
                                 + 0.935856 w_k1_MSFT_s014
                                 + 1.16142 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.925776 w_k1_ADM_s014
                                 + 0.854484 w_k1_AXP_s014
                                 + 0.95061 w_k1_DIS_s014
                                 + 0.907736 w_k1_FDX_s014
                                 + 1.07444 w_k1_IBM_s014
                                 + 0.915038 w_k1_INTC_s014
                                 + 0.994905 w_k1_KO_s014
                                 + 0.903914 w_k1_MAT_s014
                                 + 0.881851 w_k1_MSFT_s014
                                 + 1.0395 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.832567 w_k1_ADM_s014
                                 + 0.692519 w_k1_AXP_s014
                                 + 0.741359 w_k1_DIS_s014
                                 + 0.797218 w_k1_FDX_s014
                                 + 0.837068 w_k1_IBM_s014
                                 + 0.84398 w_k1_INTC_s014
                                 + 0.847288 w_k1_KO_s014
                                 + 0.678166 w_k1_MAT_s014
                                 + 0.780607 w_k1_MSFT_s014
                                 + 0.867714 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.992659 w_k1_ADM_s015
                                 + 1.47702 w_k1_AXP_s015
                                 + 1.54876 w_k1_DIS_s015
                                 + 1.34404 w_k1_FDX_s015 + 1.0448 w_k1_IBM_s015
                                 + 1.39535 w_k1_INTC_s015
                                 + 1.24993 w_k1_KO_s015 + 1.09281 w_k1_MAT_s015
                                 + 1.35759 w_k1_MSFT_s015
                                 + 1.01197 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.968188 w_k1_ADM_s015
                                 + 0.972948 w_k1_AXP_s015
                                 + 0.840308 w_k1_DIS_s015
                                 + 1.08119 w_k1_FDX_s015
                                 + 0.756204 w_k1_IBM_s015
                                 + 0.983293 w_k1_INTC_s015
                                 + 0.991437 w_k1_KO_s015
                                 + 0.784295 w_k1_MAT_s015
                                 + 0.987258 w_k1_MSFT_s015
                                 + 0.964936 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.785741 w_k1_ADM_s015
                                 + 0.76878 w_k1_AXP_s015
                                 + 0.652847 w_k1_DIS_s015
                                 + 0.85544 w_k1_FDX_s015
                                 + 0.580037 w_k1_IBM_s015
                                 + 0.819743 w_k1_INTC_s015
                                 + 0.687746 w_k1_KO_s015
                                 + 0.671295 w_k1_MAT_s015
                                 + 0.826097 w_k1_MSFT_s015
                                 + 0.788216 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.987476 w_k1_ADM_s015
                                 + 0.992332 w_k1_AXP_s015
                                 + 1.08259 w_k1_DIS_s015
                                 + 1.24305 w_k1_FDX_s015
                                 + 0.739477 w_k1_IBM_s015
                                 + 1.0664 w_k1_INTC_s015 + 1.10984 w_k1_KO_s015
                                 + 0.634927 w_k1_MAT_s015
                                 + 1.11896 w_k1_MSFT_s015
                                 + 0.880556 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.00843 w_k1_ADM_s015
                                 + 0.953129 w_k1_AXP_s015
                                 + 1.0594 w_k1_DIS_s015 + 1.15655 w_k1_FDX_s015
                                 + 1.01322 w_k1_IBM_s015
                                 + 0.897258 w_k1_INTC_s015
                                 + 0.997986 w_k1_KO_s015
                                 + 0.920346 w_k1_MAT_s015
                                 + 1.07305 w_k1_MSFT_s015
                                 + 0.897974 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.10679 w_k1_ADM_s015
                                 + 1.02761 w_k1_AXP_s015
                                 + 1.13027 w_k1_DIS_s015
                                 + 1.01997 w_k1_FDX_s015
                                 + 1.53978 w_k1_IBM_s015
                                 + 1.08342 w_k1_INTC_s015
                                 + 1.21554 w_k1_KO_s015 + 1.43834 w_k1_MAT_s015
                                 + 1.05209 w_k1_MSFT_s015
                                 + 1.04303 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.809885 w_k1_ADM_s015
                                 + 0.849512 w_k1_AXP_s015
                                 + 0.673627 w_k1_DIS_s015
                                 + 0.832864 w_k1_FDX_s015
                                 + 0.964413 w_k1_IBM_s015
                                 + 0.911721 w_k1_INTC_s015
                                 + 0.916053 w_k1_KO_s015
                                 + 0.866662 w_k1_MAT_s015
                                 + 0.926995 w_k1_MSFT_s015
                                 + 0.827059 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.900719 w_k1_ADM_s015
                                 + 0.76822 w_k1_AXP_s015
                                 + 0.82882 w_k1_DIS_s015
                                 + 0.893653 w_k1_FDX_s015
                                 + 0.766444 w_k1_IBM_s015
                                 + 0.989904 w_k1_INTC_s015
                                 + 0.884161 w_k1_KO_s015
                                 + 0.534147 w_k1_MAT_s015
                                 + 1.04415 w_k1_MSFT_s015
                                 + 0.734328 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.08676 w_k1_ADM_s016
                                 + 0.980067 w_k1_AXP_s016
                                 + 1.01163 w_k1_DIS_s016
                                 + 1.09468 w_k1_FDX_s016
                                 + 0.824993 w_k1_IBM_s016
                                 + 1.11636 w_k1_INTC_s016
                                 + 0.954296 w_k1_KO_s016
                                 + 0.914532 w_k1_MAT_s016
                                 + 1.10565 w_k1_MSFT_s016
                                 + 1.07322 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.17681 w_k1_ADM_s016
                                 + 1.0422 w_k1_AXP_s016 + 1.00145 w_k1_DIS_s016
                                 + 1.06032 w_k1_FDX_s016
                                 + 1.38345 w_k1_IBM_s016
                                 + 0.953942 w_k1_INTC_s016
                                 + 1.2185 w_k1_KO_s016 + 1.37618 w_k1_MAT_s016
                                 + 1.16079 w_k1_MSFT_s016
                                 + 1.11015 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.853291 w_k1_ADM_s016
                                 + 0.71406 w_k1_AXP_s016
                                 + 0.732169 w_k1_DIS_s016
                                 + 0.841718 w_k1_FDX_s016
                                 + 0.831786 w_k1_IBM_s016
                                 + 1.01785 w_k1_INTC_s016
                                 + 0.844921 w_k1_KO_s016
                                 + 0.777617 w_k1_MAT_s016
                                 + 0.948546 w_k1_MSFT_s016
                                 + 0.852914 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.15138 w_k1_ADM_s016
                                 + 1.28453 w_k1_AXP_s016 + 1.5867 w_k1_DIS_s016
                                 + 1.21202 w_k1_FDX_s016
                                 + 1.20944 w_k1_IBM_s016
                                 + 1.18493 w_k1_INTC_s016
                                 + 1.49704 w_k1_KO_s016 + 1.53785 w_k1_MAT_s016
                                 + 1.27674 w_k1_MSFT_s016
                                 + 1.37848 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.89841 w_k1_ADM_s016
                                 + 0.765496 w_k1_AXP_s016
                                 + 1.00862 w_k1_DIS_s016
                                 + 0.941393 w_k1_FDX_s016
                                 + 0.790657 w_k1_IBM_s016
                                 + 1.15635 w_k1_INTC_s016
                                 + 0.91692 w_k1_KO_s016
                                 + 0.925073 w_k1_MAT_s016
                                 + 1.16679 w_k1_MSFT_s016
                                 + 1.06986 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.19514 w_k1_ADM_s016
                                 + 1.05679 w_k1_AXP_s016 + 1.2096 w_k1_DIS_s016
                                 + 1.13169 w_k1_FDX_s016
                                 + 1.16553 w_k1_IBM_s016
                                 + 1.21696 w_k1_INTC_s016
                                 + 1.24507 w_k1_KO_s016 + 1.06789 w_k1_MAT_s016
                                 + 1.16511 w_k1_MSFT_s016
                                 + 1.17176 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.2702 w_k1_ADM_s016
                                 + 1.07507 w_k1_AXP_s016
                                 + 1.03135 w_k1_DIS_s016
                                 + 1.02637 w_k1_FDX_s016
                                 + 1.02287 w_k1_IBM_s016
                                 + 0.965112 w_k1_INTC_s016
                                 + 1.18181 w_k1_KO_s016 + 1.33089 w_k1_MAT_s016
                                 + 1.12798 w_k1_MSFT_s016
                                 + 1.15628 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.2931 w_k1_ADM_s016
                                 + 1.13465 w_k1_AXP_s016
                                 + 1.13127 w_k1_DIS_s016
                                 + 1.24352 w_k1_FDX_s016
                                 + 0.777349 w_k1_IBM_s016
                                 + 1.10024 w_k1_INTC_s016
                                 + 0.985494 w_k1_KO_s016
                                 + 0.890209 w_k1_MAT_s016
                                 + 1.14782 w_k1_MSFT_s016
                                 + 1.05848 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.05023 w_k1_ADM_s017
                                 + 1.00626 w_k1_AXP_s017
                                 + 0.878747 w_k1_DIS_s017
                                 + 0.921503 w_k1_FDX_s017
                                 + 0.931713 w_k1_IBM_s017
                                 + 0.971224 w_k1_INTC_s017
                                 + 1.04881 w_k1_KO_s017 + 1.12839 w_k1_MAT_s017
                                 + 0.985317 w_k1_MSFT_s017
                                 + 0.918212 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.908713 w_k1_ADM_s017
                                 + 1.05846 w_k1_AXP_s017
                                 + 1.23627 w_k1_DIS_s017
                                 + 1.02466 w_k1_FDX_s017
                                 + 1.13239 w_k1_IBM_s017
                                 + 1.0017 w_k1_INTC_s017 + 1.23489 w_k1_KO_s017
                                 + 1.16801 w_k1_MAT_s017
                                 + 1.11916 w_k1_MSFT_s017
                                 + 1.13981 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.756141 w_k1_ADM_s017
                                 + 0.637215 w_k1_AXP_s017
                                 + 0.618894 w_k1_DIS_s017
                                 + 0.725852 w_k1_FDX_s017
                                 + 0.720515 w_k1_IBM_s017
                                 + 0.675565 w_k1_INTC_s017
                                 + 0.731867 w_k1_KO_s017
                                 + 0.716186 w_k1_MAT_s017
                                 + 0.820732 w_k1_MSFT_s017
                                 + 0.819551 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.1099 w_k1_ADM_s017
                                 + 1.2071 w_k1_AXP_s017 + 1.44579 w_k1_DIS_s017
                                 + 1.12579 w_k1_FDX_s017
                                 + 1.59271 w_k1_IBM_s017
                                 + 1.25963 w_k1_INTC_s017
                                 + 1.50644 w_k1_KO_s017 + 1.70474 w_k1_MAT_s017
                                 + 1.00932 w_k1_MSFT_s017
                                 + 1.25004 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.39626 w_k1_ADM_s017
                                 + 1.16457 w_k1_AXP_s017
                                 + 1.57569 w_k1_DIS_s017
                                 + 1.34379 w_k1_FDX_s017
                                 + 1.01041 w_k1_IBM_s017
                                 + 1.14501 w_k1_INTC_s017 + 1.1776 w_k1_KO_s017
                                 + 0.898571 w_k1_MAT_s017
                                 + 1.16071 w_k1_MSFT_s017
                                 + 1.27388 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.985112 w_k1_ADM_s017
                                 + 0.6951 w_k1_AXP_s017
                                 + 0.881013 w_k1_DIS_s017
                                 + 0.771641 w_k1_FDX_s017
                                 + 0.967537 w_k1_IBM_s017
                                 + 0.88154 w_k1_INTC_s017
                                 + 1.09463 w_k1_KO_s017
                                 + 0.857559 w_k1_MAT_s017
                                 + 0.851413 w_k1_MSFT_s017
                                 + 0.933628 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.985445 w_k1_ADM_s017
                                 + 1.34995 w_k1_AXP_s017
                                 + 1.20413 w_k1_DIS_s017
                                 + 1.08964 w_k1_FDX_s017
                                 + 1.35013 w_k1_IBM_s017
                                 + 1.15967 w_k1_INTC_s017
                                 + 1.43018 w_k1_KO_s017 + 1.79374 w_k1_MAT_s017
                                 + 1.22187 w_k1_MSFT_s017 + 1.1117 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.01098 w_k1_ADM_s017
                                 + 0.729305 w_k1_AXP_s017
                                 + 0.926472 w_k1_DIS_s017
                                 + 0.926829 w_k1_FDX_s017
                                 + 0.665347 w_k1_IBM_s017
                                 + 0.812109 w_k1_INTC_s017
                                 + 0.833832 w_k1_KO_s017
                                 + 0.545467 w_k1_MAT_s017
                                 + 0.931574 w_k1_MSFT_s017
                                 + 0.940705 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.861872 w_k1_ADM_s100
                                 + 0.850672 w_k1_AXP_s100
                                 + 0.938085 w_k1_DIS_s100
                                 + 0.920656 w_k1_FDX_s100
                                 + 0.991726 w_k1_IBM_s100
                                 + 0.826962 w_k1_INTC_s100
                                 + 0.874366 w_k1_KO_s100
                                 + 1.56027 w_k1_MAT_s100
                                 + 0.824581 w_k1_MSFT_s100
                                 + 0.953422 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.9437 w_k1_ADM_s100
                                 + 0.709217 w_k1_AXP_s100
                                 + 0.563922 w_k1_DIS_s100
                                 + 0.681607 w_k1_FDX_s100
                                 + 0.931447 w_k1_IBM_s100
                                 + 0.848355 w_k1_INTC_s100
                                 + 0.813798 w_k1_KO_s100
                                 + 0.90346 w_k1_MAT_s100
                                 + 0.689707 w_k1_MSFT_s100
                                 + 0.846342 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1932 w_k1_ADM_s100
                                 + 1.65584 w_k1_AXP_s100
                                 + 1.52454 w_k1_DIS_s100
                                 + 1.33209 w_k1_FDX_s100
                                 + 1.18085 w_k1_IBM_s100
                                 + 1.10987 w_k1_INTC_s100 + 1.1625 w_k1_KO_s100
                                 + 1.71016 w_k1_MAT_s100
                                 + 1.14881 w_k1_MSFT_s100
                                 + 1.21797 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.731214 w_k1_ADM_s100
                                 + 0.643244 w_k1_AXP_s100
                                 + 0.955361 w_k1_DIS_s100
                                 + 0.788791 w_k1_FDX_s100
                                 + 1.02451 w_k1_IBM_s100
                                 + 0.914405 w_k1_INTC_s100
                                 + 0.933833 w_k1_KO_s100
                                 + 0.75411 w_k1_MAT_s100
                                 + 0.803996 w_k1_MSFT_s100
                                 + 0.871911 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.13202 w_k1_ADM_s100
                                 + 0.979967 w_k1_AXP_s100
                                 + 1.03793 w_k1_DIS_s100 + 1.0281 w_k1_FDX_s100
                                 + 0.820996 w_k1_IBM_s100
                                 + 0.901369 w_k1_INTC_s100
                                 + 0.953879 w_k1_KO_s100
                                 + 1.08396 w_k1_MAT_s100
                                 + 0.944945 w_k1_MSFT_s100
                                 + 0.965532 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.840063 w_k1_ADM_s100
                                 + 0.943514 w_k1_AXP_s100
                                 + 1.12487 w_k1_DIS_s100
                                 + 0.903626 w_k1_FDX_s100
                                 + 1.39951 w_k1_IBM_s100
                                 + 0.989525 w_k1_INTC_s100
                                 + 1.05779 w_k1_KO_s100 + 1.41838 w_k1_MAT_s100
                                 + 0.886837 w_k1_MSFT_s100
                                 + 0.987895 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.973548 w_k1_ADM_s100
                                 + 0.772098 w_k1_AXP_s100
                                 + 0.814474 w_k1_DIS_s100
                                 + 0.816887 w_k1_FDX_s100
                                 + 0.721897 w_k1_IBM_s100
                                 + 0.877163 w_k1_INTC_s100
                                 + 0.861731 w_k1_KO_s100
                                 + 0.761491 w_k1_MAT_s100
                                 + 0.792814 w_k1_MSFT_s100
                                 + 0.901675 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.0688 w_k1_ADM_s100
                                 + 1.09979 w_k1_AXP_s100 + 1.4455 w_k1_DIS_s100
                                 + 1.3393 w_k1_FDX_s100
                                 + 0.821736 w_k1_IBM_s100
                                 + 1.27493 w_k1_INTC_s100
                                 + 1.05825 w_k1_KO_s100 + 0.87112 w_k1_MAT_s100
                                 + 1.13548 w_k1_MSFT_s100
                                 + 1.17781 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.939959 w_k1_ADM_s101
                                 + 0.6679 w_k1_AXP_s101
                                 + 0.681085 w_k1_DIS_s101
                                 + 0.80985 w_k1_FDX_s101
                                 + 1.13537 w_k1_IBM_s101
                                 + 0.941698 w_k1_INTC_s101
                                 + 0.964758 w_k1_KO_s101
                                 + 1.36314 w_k1_MAT_s101
                                 + 0.906728 w_k1_MSFT_s101
                                 + 0.841711 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.941975 w_k1_ADM_s101
                                 + 0.683753 w_k1_AXP_s101
                                 + 0.877622 w_k1_DIS_s101
                                 + 0.92525 w_k1_FDX_s101
                                 + 0.761309 w_k1_IBM_s101
                                 + 1.00176 w_k1_INTC_s101
                                 + 0.90965 w_k1_KO_s101
                                 + 0.998779 w_k1_MAT_s101
                                 + 1.04838 w_k1_MSFT_s101
                                 + 0.92422 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.23589 w_k1_ADM_s101
                                 + 1.45579 w_k1_AXP_s101
                                 + 1.67924 w_k1_DIS_s101
                                 + 1.39611 w_k1_FDX_s101
                                 + 1.67088 w_k1_IBM_s101
                                 + 1.28884 w_k1_INTC_s101
                                 + 1.46405 w_k1_KO_s101 + 2.36205 w_k1_MAT_s101
                                 + 1.48872 w_k1_MSFT_s101
                                 + 1.21956 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.973518 w_k1_ADM_s101
                                 + 1.51254 w_k1_AXP_s101
                                 + 1.20736 w_k1_DIS_s101
                                 + 1.24011 w_k1_FDX_s101
                                 + 1.28089 w_k1_IBM_s101
                                 + 1.08041 w_k1_INTC_s101
                                 + 1.10824 w_k1_KO_s101 + 2.16992 w_k1_MAT_s101
                                 + 1.22122 w_k1_MSFT_s101
                                 + 1.12195 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.23385 w_k1_ADM_s101
                                 + 1.19843 w_k1_AXP_s101
                                 + 0.832486 w_k1_DIS_s101
                                 + 1.07555 w_k1_FDX_s101
                                 + 0.888246 w_k1_IBM_s101
                                 + 1.07621 w_k1_INTC_s101
                                 + 0.83888 w_k1_KO_s101 + 1.52923 w_k1_MAT_s101
                                 + 0.941197 w_k1_MSFT_s101
                                 + 0.819162 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.21373 w_k1_ADM_s101
                                 + 1.06811 w_k1_AXP_s101
                                 + 1.16463 w_k1_DIS_s101
                                 + 1.36934 w_k1_FDX_s101
                                 + 0.803744 w_k1_IBM_s101
                                 + 1.33823 w_k1_INTC_s101
                                 + 0.872268 w_k1_KO_s101
                                 + 0.866023 w_k1_MAT_s101
                                 + 1.17354 w_k1_MSFT_s101
                                 + 0.937104 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.09931 w_k1_ADM_s101
                                 + 0.86727 w_k1_AXP_s101
                                 + 1.07363 w_k1_DIS_s101
                                 + 0.998924 w_k1_FDX_s101
                                 + 1.42566 w_k1_IBM_s101
                                 + 1.01333 w_k1_INTC_s101
                                 + 1.25969 w_k1_KO_s101 + 1.23615 w_k1_MAT_s101
                                 + 1.05256 w_k1_MSFT_s101
                                 + 1.01145 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.24559 w_k1_ADM_s101
                                 + 1.22803 w_k1_AXP_s101
                                 + 1.32234 w_k1_DIS_s101
                                 + 1.30982 w_k1_FDX_s101
                                 + 0.878285 w_k1_IBM_s101
                                 + 1.24077 w_k1_INTC_s101
                                 + 1.01185 w_k1_KO_s101 + 1.52749 w_k1_MAT_s101
                                 + 1.15664 w_k1_MSFT_s101
                                 + 1.03989 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.00106 w_k1_ADM_s102
                                 + 0.958971 w_k1_AXP_s102
                                 + 0.832274 w_k1_DIS_s102
                                 + 1.06461 w_k1_FDX_s102
                                 + 0.636691 w_k1_IBM_s102
                                 + 0.86992 w_k1_INTC_s102
                                 + 0.705142 w_k1_KO_s102
                                 + 0.757361 w_k1_MAT_s102
                                 + 0.929702 w_k1_MSFT_s102
                                 + 1.0259 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.871857 w_k1_ADM_s102
                                 + 1.15502 w_k1_AXP_s102
                                 + 1.01481 w_k1_DIS_s102
                                 + 1.00844 w_k1_FDX_s102
                                 + 1.17566 w_k1_IBM_s102
                                 + 1.01198 w_k1_INTC_s102
                                 + 0.895421 w_k1_KO_s102
                                 + 1.59522 w_k1_MAT_s102
                                 + 1.06958 w_k1_MSFT_s102
                                 + 1.09203 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.913007 w_k1_ADM_s102
                                 + 0.863621 w_k1_AXP_s102
                                 + 1.05095 w_k1_DIS_s102
                                 + 1.01296 w_k1_FDX_s102
                                 + 0.938632 w_k1_IBM_s102
                                 + 1.06695 w_k1_INTC_s102
                                 + 0.912963 w_k1_KO_s102
                                 + 0.934707 w_k1_MAT_s102
                                 + 1.08393 w_k1_MSFT_s102
                                 + 1.01833 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.14859 w_k1_ADM_s102
                                 + 1.69499 w_k1_AXP_s102
                                 + 1.39046 w_k1_DIS_s102
                                 + 1.22164 w_k1_FDX_s102
                                 + 1.47974 w_k1_IBM_s102
                                 + 1.37982 w_k1_INTC_s102
                                 + 1.36549 w_k1_KO_s102 + 2.68787 w_k1_MAT_s102
                                 + 1.14961 w_k1_MSFT_s102
                                 + 1.33422 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.01599 w_k1_ADM_s102
                                 + 1.04876 w_k1_AXP_s102
                                 + 0.724392 w_k1_DIS_s102
                                 + 0.842436 w_k1_FDX_s102
                                 + 0.977781 w_k1_IBM_s102
                                 + 0.825348 w_k1_INTC_s102
                                 + 0.844883 w_k1_KO_s102 + 1.4364 w_k1_MAT_s102
                                 + 0.863146 w_k1_MSFT_s102
                                 + 1.02239 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.747245 w_k1_ADM_s102
                                 + 0.735866 w_k1_AXP_s102
                                 + 0.605381 w_k1_DIS_s102
                                 + 0.786119 w_k1_FDX_s102
                                 + 0.710448 w_k1_IBM_s102
                                 + 0.855459 w_k1_INTC_s102
                                 + 0.646681 w_k1_KO_s102
                                 + 0.775425 w_k1_MAT_s102
                                 + 0.768989 w_k1_MSFT_s102
                                 + 0.928052 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.524398 w_k1_ADM_s102
                                 + 0.595175 w_k1_AXP_s102
                                 + 0.573117 w_k1_DIS_s102
                                 + 0.688896 w_k1_FDX_s102
                                 + 0.624705 w_k1_IBM_s102
                                 + 0.693433 w_k1_INTC_s102
                                 + 0.518129 w_k1_KO_s102
                                 + 0.826515 w_k1_MAT_s102
                                 + 0.682807 w_k1_MSFT_s102
                                 + 0.767355 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.21208 w_k1_ADM_s102
                                 + 1.14722 w_k1_AXP_s102
                                 + 1.20225 w_k1_DIS_s102
                                 + 1.17782 w_k1_FDX_s102
                                 + 0.789311 w_k1_IBM_s102
                                 + 1.07603 w_k1_INTC_s102
                                 + 0.855027 w_k1_KO_s102
                                 + 1.00411 w_k1_MAT_s102
                                 + 1.06092 w_k1_MSFT_s102
                                 + 1.06063 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.838146 w_k1_ADM_s103
                                 + 1.16578 w_k1_AXP_s103
                                 + 0.970662 w_k1_DIS_s103
                                 + 0.968418 w_k1_FDX_s103
                                 + 0.886701 w_k1_IBM_s103
                                 + 1.20929 w_k1_INTC_s103
                                 + 0.90781 w_k1_KO_s103 + 1.11563 w_k1_MAT_s103
                                 + 0.944388 w_k1_MSFT_s103
                                 + 0.967124 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.755679 w_k1_ADM_s103
                                 + 0.672945 w_k1_AXP_s103
                                 + 0.789428 w_k1_DIS_s103
                                 + 0.869423 w_k1_FDX_s103
                                 + 0.684738 w_k1_IBM_s103
                                 + 0.960303 w_k1_INTC_s103
                                 + 0.642052 w_k1_KO_s103
                                 + 0.67493 w_k1_MAT_s103
                                 + 0.900619 w_k1_MSFT_s103
                                 + 0.854564 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.27502 w_k1_ADM_s103
                                 + 1.23784 w_k1_AXP_s103
                                 + 1.34029 w_k1_DIS_s103 + 1.1303 w_k1_FDX_s103
                                 + 1.38785 w_k1_IBM_s103
                                 + 1.14842 w_k1_INTC_s103
                                 + 1.15471 w_k1_KO_s103 + 1.35776 w_k1_MAT_s103
                                 + 1.18933 w_k1_MSFT_s103
                                 + 1.17228 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.879973 w_k1_ADM_s103
                                 + 0.878741 w_k1_AXP_s103
                                 + 1.00065 w_k1_DIS_s103
                                 + 1.06729 w_k1_FDX_s103
                                 + 0.976193 w_k1_IBM_s103
                                 + 1.03007 w_k1_INTC_s103
                                 + 0.885475 w_k1_KO_s103
                                 + 0.710073 w_k1_MAT_s103
                                 + 0.953683 w_k1_MSFT_s103
                                 + 1.00085 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.757049 w_k1_ADM_s103
                                 + 0.729062 w_k1_AXP_s103
                                 + 0.629285 w_k1_DIS_s103
                                 + 0.765623 w_k1_FDX_s103
                                 + 0.865654 w_k1_IBM_s103
                                 + 0.737765 w_k1_INTC_s103
                                 + 0.656024 w_k1_KO_s103
                                 + 0.753186 w_k1_MAT_s103
                                 + 0.67676 w_k1_MSFT_s103
                                 + 0.795906 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.903672 w_k1_ADM_s103
                                 + 0.717059 w_k1_AXP_s103
                                 + 0.882061 w_k1_DIS_s103
                                 + 0.84584 w_k1_FDX_s103
                                 + 1.02398 w_k1_IBM_s103
                                 + 1.02025 w_k1_INTC_s103
                                 + 0.827835 w_k1_KO_s103
                                 + 0.878586 w_k1_MAT_s103
                                 + 0.936111 w_k1_MSFT_s103
                                 + 1.00037 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.990159 w_k1_ADM_s103
                                 + 0.960518 w_k1_AXP_s103
                                 + 0.829513 w_k1_DIS_s103
                                 + 1.03383 w_k1_FDX_s103
                                 + 0.63741 w_k1_IBM_s103
                                 + 0.929633 w_k1_INTC_s103
                                 + 0.639818 w_k1_KO_s103
                                 + 0.415299 w_k1_MAT_s103
                                 + 0.936488 w_k1_MSFT_s103
                                 + 0.84054 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.16231 w_k1_ADM_s103
                                 + 1.02125 w_k1_AXP_s103
                                 + 1.56855 w_k1_DIS_s103
                                 + 1.26071 w_k1_FDX_s103
                                 + 0.898045 w_k1_IBM_s103
                                 + 0.994018 w_k1_INTC_s103
                                 + 1.03351 w_k1_KO_s103
                                 + 0.644226 w_k1_MAT_s103
                                 + 1.31136 w_k1_MSFT_s103
                                 + 1.12471 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.17666 w_k1_ADM_s104
                                 + 1.21838 w_k1_AXP_s104
                                 + 1.22062 w_k1_DIS_s104
                                 + 1.10859 w_k1_FDX_s104
                                 + 1.02712 w_k1_IBM_s104
                                 + 1.07024 w_k1_INTC_s104
                                 + 1.01205 w_k1_KO_s104
                                 + 0.963221 w_k1_MAT_s104
                                 + 1.06284 w_k1_MSFT_s104
                                 + 1.14229 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.992749 w_k1_ADM_s104
                                 + 1.24178 w_k1_AXP_s104
                                 + 0.757786 w_k1_DIS_s104
                                 + 0.882693 w_k1_FDX_s104
                                 + 1.43149 w_k1_IBM_s104
                                 + 1.05301 w_k1_INTC_s104
                                 + 1.02431 w_k1_KO_s104 + 1.549 w_k1_MAT_s104
                                 + 0.950883 w_k1_MSFT_s104
                                 + 1.0362 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.40891 w_k1_ADM_s104
                                 + 1.80856 w_k1_AXP_s104
                                 + 1.46086 w_k1_DIS_s104
                                 + 1.40675 w_k1_FDX_s104 + 1.2439 w_k1_IBM_s104
                                 + 1.18029 w_k1_INTC_s104
                                 + 1.31032 w_k1_KO_s104
                                 + 0.993873 w_k1_MAT_s104
                                 + 1.28571 w_k1_MSFT_s104
                                 + 1.29823 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.85903 w_k1_ADM_s104
                                 + 0.896034 w_k1_AXP_s104
                                 + 0.72934 w_k1_DIS_s104
                                 + 0.817374 w_k1_FDX_s104
                                 + 1.05481 w_k1_IBM_s104
                                 + 0.816062 w_k1_INTC_s104
                                 + 0.828605 w_k1_KO_s104
                                 + 0.981097 w_k1_MAT_s104
                                 + 0.83774 w_k1_MSFT_s104
                                 + 0.892571 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.21684 w_k1_ADM_s104
                                 + 1.26938 w_k1_AXP_s104
                                 + 1.61342 w_k1_DIS_s104
                                 + 1.18986 w_k1_FDX_s104
                                 + 1.47086 w_k1_IBM_s104
                                 + 1.28754 w_k1_INTC_s104
                                 + 1.52695 w_k1_KO_s104 + 1.02717 w_k1_MAT_s104
                                 + 1.28523 w_k1_MSFT_s104
                                 + 1.16398 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.93829 w_k1_ADM_s104
                                 + 1.02841 w_k1_AXP_s104
                                 + 0.969996 w_k1_DIS_s104
                                 + 0.991769 w_k1_FDX_s104
                                 + 0.873158 w_k1_IBM_s104
                                 + 1.07955 w_k1_INTC_s104
                                 + 0.937335 w_k1_KO_s104
                                 + 0.895174 w_k1_MAT_s104
                                 + 1.01071 w_k1_MSFT_s104
                                 + 1.02473 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.906495 w_k1_ADM_s104
                                 + 0.792845 w_k1_AXP_s104
                                 + 0.756032 w_k1_DIS_s104
                                 + 0.865606 w_k1_FDX_s104
                                 + 0.69813 w_k1_IBM_s104
                                 + 0.791642 w_k1_INTC_s104
                                 + 0.717942 w_k1_KO_s104
                                 + 0.501439 w_k1_MAT_s104
                                 + 0.834152 w_k1_MSFT_s104
                                 + 0.905423 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.934009 w_k1_ADM_s104
                                 + 1.42784 w_k1_AXP_s104
                                 + 0.990066 w_k1_DIS_s104
                                 + 1.08016 w_k1_FDX_s104 + 1.1814 w_k1_IBM_s104
                                 + 1.06308 w_k1_INTC_s104
                                 + 0.995486 w_k1_KO_s104 + 1.5664 w_k1_MAT_s104
                                 + 1.02809 w_k1_MSFT_s104
                                 + 1.10907 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.972612 w_k1_ADM_s105
                                 + 1.15247 w_k1_AXP_s105
                                 + 1.22879 w_k1_DIS_s105
                                 + 1.11735 w_k1_FDX_s105
                                 + 0.899019 w_k1_IBM_s105
                                 + 1.11572 w_k1_INTC_s105
                                 + 1.09702 w_k1_KO_s105 + 1.07709 w_k1_MAT_s105
                                 + 0.973883 w_k1_MSFT_s105
                                 + 1.04881 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.975569 w_k1_ADM_s105
                                 + 0.905846 w_k1_AXP_s105
                                 + 1.0006 w_k1_DIS_s105
                                 + 0.988726 w_k1_FDX_s105
                                 + 0.801646 w_k1_IBM_s105
                                 + 0.979377 w_k1_INTC_s105
                                 + 1.03256 w_k1_KO_s105 + 1.0716 w_k1_MAT_s105
                                 + 0.95197 w_k1_MSFT_s105
                                 + 1.05658 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.944268 w_k1_ADM_s105
                                 + 1.45337 w_k1_AXP_s105
                                 + 1.34089 w_k1_DIS_s105
                                 + 1.10264 w_k1_FDX_s105
                                 + 1.41764 w_k1_IBM_s105
                                 + 1.10317 w_k1_INTC_s105
                                 + 1.34257 w_k1_KO_s105 + 2.07135 w_k1_MAT_s105
                                 + 0.997026 w_k1_MSFT_s105
                                 + 1.14688 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.853375 w_k1_ADM_s105
                                 + 1.0861 w_k1_AXP_s105 + 1.05674 w_k1_DIS_s105
                                 + 0.849942 w_k1_FDX_s105
                                 + 1.35468 w_k1_IBM_s105
                                 + 0.737095 w_k1_INTC_s105
                                 + 1.19178 w_k1_KO_s105 + 1.49556 w_k1_MAT_s105
                                 + 0.843898 w_k1_MSFT_s105
                                 + 1.03217 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.837653 w_k1_ADM_s105
                                 + 0.897795 w_k1_AXP_s105
                                 + 0.684762 w_k1_DIS_s105
                                 + 0.843347 w_k1_FDX_s105
                                 + 0.980836 w_k1_IBM_s105
                                 + 0.869731 w_k1_INTC_s105
                                 + 0.81911 w_k1_KO_s105 + 1.34254 w_k1_MAT_s105
                                 + 0.730148 w_k1_MSFT_s105
                                 + 0.939105 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.713663 w_k1_ADM_s105
                                 + 0.700655 w_k1_AXP_s105
                                 + 0.566103 w_k1_DIS_s105
                                 + 0.670087 w_k1_FDX_s105
                                 + 0.682434 w_k1_IBM_s105
                                 + 0.692413 w_k1_INTC_s105
                                 + 0.640527 w_k1_KO_s105
                                 + 0.692003 w_k1_MAT_s105
                                 + 0.720828 w_k1_MSFT_s105
                                 + 0.83653 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.830358 w_k1_ADM_s105
                                 + 0.732358 w_k1_AXP_s105
                                 + 0.659967 w_k1_DIS_s105
                                 + 0.851473 w_k1_FDX_s105
                                 + 0.577534 w_k1_IBM_s105
                                 + 0.813725 w_k1_INTC_s105
                                 + 0.692693 w_k1_KO_s105
                                 + 0.601613 w_k1_MAT_s105
                                 + 0.81866 w_k1_MSFT_s105
                                 + 0.843172 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.22104 w_k1_ADM_s105
                                 + 1.33223 w_k1_AXP_s105
                                 + 1.53086 w_k1_DIS_s105
                                 + 1.25339 w_k1_FDX_s105
                                 + 1.13517 w_k1_IBM_s105
                                 + 0.952512 w_k1_INTC_s105
                                 + 1.28505 w_k1_KO_s105 + 1.57923 w_k1_MAT_s105
                                 + 1.09917 w_k1_MSFT_s105
                                 + 1.21452 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.18369 w_k1_ADM_s106
                                 + 1.11215 w_k1_AXP_s106
                                 + 1.19424 w_k1_DIS_s106
                                 + 1.03859 w_k1_FDX_s106 + 1.0994 w_k1_IBM_s106
                                 + 1.05248 w_k1_INTC_s106
                                 + 1.13466 w_k1_KO_s106 + 1.15367 w_k1_MAT_s106
                                 + 1.14906 w_k1_MSFT_s106
                                 + 1.00232 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.832301 w_k1_ADM_s106
                                 + 0.583868 w_k1_AXP_s106
                                 + 0.55232 w_k1_DIS_s106
                                 + 0.728429 w_k1_FDX_s106
                                 + 0.591656 w_k1_IBM_s106
                                 + 0.619071 w_k1_INTC_s106
                                 + 0.653046 w_k1_KO_s106
                                 + 0.61671 w_k1_MAT_s106
                                 + 0.729785 w_k1_MSFT_s106
                                 + 0.785959 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.32834 w_k1_ADM_s106
                                 + 1.09599 w_k1_AXP_s106
                                 + 1.46302 w_k1_DIS_s106
                                 + 1.33791 w_k1_FDX_s106
                                 + 0.806769 w_k1_IBM_s106
                                 + 1.173 w_k1_INTC_s106 + 1.132 w_k1_KO_s106
                                 + 1.26098 w_k1_MAT_s106
                                 + 1.18187 w_k1_MSFT_s106
                                 + 1.14014 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.07552 w_k1_ADM_s106
                                 + 0.837618 w_k1_AXP_s106
                                 + 1.13819 w_k1_DIS_s106
                                 + 0.932186 w_k1_FDX_s106
                                 + 1.41081 w_k1_IBM_s106
                                 + 0.871018 w_k1_INTC_s106
                                 + 1.29881 w_k1_KO_s106 + 1.74738 w_k1_MAT_s106
                                 + 1.09379 w_k1_MSFT_s106
                                 + 1.08184 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.1252 w_k1_ADM_s106
                                 + 0.906636 w_k1_AXP_s106
                                 + 1.02223 w_k1_DIS_s106
                                 + 1.08699 w_k1_FDX_s106
                                 + 0.97202 w_k1_IBM_s106
                                 + 0.930893 w_k1_INTC_s106
                                 + 1.01163 w_k1_KO_s106
                                 + 0.963775 w_k1_MAT_s106
                                 + 0.946429 w_k1_MSFT_s106
                                 + 0.954912 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.43699 w_k1_ADM_s106
                                 + 1.38058 w_k1_AXP_s106
                                 + 1.50156 w_k1_DIS_s106
                                 + 1.15991 w_k1_FDX_s106
                                 + 1.62367 w_k1_IBM_s106
                                 + 1.04111 w_k1_INTC_s106
                                 + 1.53129 w_k1_KO_s106 + 2.04427 w_k1_MAT_s106
                                 + 1.153 w_k1_MSFT_s106 + 1.13795 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.920606 w_k1_ADM_s106
                                 + 0.801761 w_k1_AXP_s106
                                 + 0.75527 w_k1_DIS_s106
                                 + 0.847407 w_k1_FDX_s106
                                 + 0.936472 w_k1_IBM_s106
                                 + 0.669021 w_k1_INTC_s106
                                 + 0.933603 w_k1_KO_s106 + 1.3087 w_k1_MAT_s106
                                 + 0.772197 w_k1_MSFT_s106
                                 + 0.874794 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.07533 w_k1_ADM_s106
                                 + 0.743903 w_k1_AXP_s106
                                 + 0.842976 w_k1_DIS_s106
                                 + 0.957585 w_k1_FDX_s106
                                 + 0.653636 w_k1_IBM_s106
                                 + 0.768273 w_k1_INTC_s106
                                 + 0.774508 w_k1_KO_s106
                                 + 0.75922 w_k1_MAT_s106
                                 + 0.868239 w_k1_MSFT_s106
                                 + 0.859479 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.783175 w_k1_ADM_s107
                                 + 0.821617 w_k1_AXP_s107
                                 + 0.783966 w_k1_DIS_s107
                                 + 0.794034 w_k1_FDX_s107
                                 + 0.736953 w_k1_IBM_s107
                                 + 0.872521 w_k1_INTC_s107
                                 + 0.888025 w_k1_KO_s107
                                 + 0.621966 w_k1_MAT_s107
                                 + 0.765726 w_k1_MSFT_s107
                                 + 0.900345 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.04197 w_k1_ADM_s107
                                 + 1.01127 w_k1_AXP_s107
                                 + 0.942239 w_k1_DIS_s107
                                 + 0.940522 w_k1_FDX_s107
                                 + 1.06918 w_k1_IBM_s107
                                 + 0.885568 w_k1_INTC_s107
                                 + 1.06116 w_k1_KO_s107
                                 + 0.883465 w_k1_MAT_s107
                                 + 0.973865 w_k1_MSFT_s107
                                 + 1.02797 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.996665 w_k1_ADM_s107
                                 + 1.00409 w_k1_AXP_s107
                                 + 1.08235 w_k1_DIS_s107
                                 + 1.14656 w_k1_FDX_s107
                                 + 0.702996 w_k1_IBM_s107
                                 + 1.02734 w_k1_INTC_s107
                                 + 0.924822 w_k1_KO_s107
                                 + 0.698624 w_k1_MAT_s107
                                 + 1.06031 w_k1_MSFT_s107
                                 + 1.04005 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.1819 w_k1_ADM_s107
                                 + 1.49581 w_k1_AXP_s107
                                 + 1.54084 w_k1_DIS_s107
                                 + 1.29165 w_k1_FDX_s107
                                 + 1.27929 w_k1_IBM_s107
                                 + 1.14078 w_k1_INTC_s107
                                 + 1.29847 w_k1_KO_s107 + 1.43498 w_k1_MAT_s107
                                 + 1.17681 w_k1_MSFT_s107
                                 + 1.20945 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.10453 w_k1_ADM_s107
                                 + 1.25427 w_k1_AXP_s107
                                 + 1.16547 w_k1_DIS_s107
                                 + 1.10268 w_k1_FDX_s107
                                 + 1.33075 w_k1_IBM_s107
                                 + 1.02999 w_k1_INTC_s107
                                 + 1.25071 w_k1_KO_s107 + 1.27605 w_k1_MAT_s107
                                 + 0.970602 w_k1_MSFT_s107
                                 + 1.18557 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.29028 w_k1_ADM_s107
                                 + 1.28638 w_k1_AXP_s107
                                 + 1.20909 w_k1_DIS_s107
                                 + 1.06479 w_k1_FDX_s107
                                 + 1.83823 w_k1_IBM_s107
                                 + 1.15029 w_k1_INTC_s107
                                 + 1.46428 w_k1_KO_s107 + 1.36881 w_k1_MAT_s107
                                 + 1.10547 w_k1_MSFT_s107
                                 + 1.20714 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.897471 w_k1_ADM_s107
                                 + 1.13047 w_k1_AXP_s107
                                 + 1.16218 w_k1_DIS_s107
                                 + 1.07625 w_k1_FDX_s107
                                 + 0.871471 w_k1_IBM_s107
                                 + 1.04787 w_k1_INTC_s107
                                 + 1.10849 w_k1_KO_s107
                                 + 0.892993 w_k1_MAT_s107
                                 + 1.02529 w_k1_MSFT_s107
                                 + 1.12059 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.717429 w_k1_ADM_s107
                                 + 0.947071 w_k1_AXP_s107
                                 + 0.621546 w_k1_DIS_s107
                                 + 0.729881 w_k1_FDX_s107
                                 + 0.971156 w_k1_IBM_s107
                                 + 0.66741 w_k1_INTC_s107
                                 + 0.891984 w_k1_KO_s107
                                 + 1.00405 w_k1_MAT_s107
                                 + 0.704267 w_k1_MSFT_s107
                                 + 0.871791 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.927746 w_k1_ADM_s110
                                 + 1.32801 w_k1_AXP_s110
                                 + 1.33579 w_k1_DIS_s110
                                 + 1.18491 w_k1_FDX_s110
                                 + 1.14208 w_k1_IBM_s110
                                 + 1.1485 w_k1_INTC_s110 + 1.13135 w_k1_KO_s110
                                 + 1.81316 w_k1_MAT_s110
                                 + 1.2731 w_k1_MSFT_s110 + 1.11787 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.924058 w_k1_ADM_s110
                                 + 0.783088 w_k1_AXP_s110
                                 + 0.793644 w_k1_DIS_s110
                                 + 0.851951 w_k1_FDX_s110
                                 + 0.899056 w_k1_IBM_s110
                                 + 0.958977 w_k1_INTC_s110
                                 + 0.822708 w_k1_KO_s110
                                 + 0.784075 w_k1_MAT_s110
                                 + 0.89714 w_k1_MSFT_s110
                                 + 0.976792 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.833784 w_k1_ADM_s110
                                 + 0.860099 w_k1_AXP_s110
                                 + 0.998079 w_k1_DIS_s110
                                 + 0.934171 w_k1_FDX_s110
                                 + 1.23028 w_k1_IBM_s110
                                 + 1.16547 w_k1_INTC_s110
                                 + 0.991431 w_k1_KO_s110
                                 + 0.80307 w_k1_MAT_s110
                                 + 1.03229 w_k1_MSFT_s110
                                 + 1.03555 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.949962 w_k1_ADM_s110
                                 + 0.91735 w_k1_AXP_s110
                                 + 1.06211 w_k1_DIS_s110 + 1.1061 w_k1_FDX_s110
                                 + 0.725737 w_k1_IBM_s110
                                 + 1.15429 w_k1_INTC_s110
                                 + 0.909158 w_k1_KO_s110
                                 + 0.881723 w_k1_MAT_s110
                                 + 1.13892 w_k1_MSFT_s110
                                 + 1.03155 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.872752 w_k1_ADM_s110
                                 + 1.47848 w_k1_AXP_s110
                                 + 1.57005 w_k1_DIS_s110
                                 + 1.25449 w_k1_FDX_s110
                                 + 1.54303 w_k1_IBM_s110
                                 + 1.42775 w_k1_INTC_s110
                                 + 1.34226 w_k1_KO_s110 + 1.87987 w_k1_MAT_s110
                                 + 1.20126 w_k1_MSFT_s110
                                 + 1.35208 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.16129 w_k1_ADM_s110
                                 + 1.10741 w_k1_AXP_s110
                                 + 1.09952 w_k1_DIS_s110 + 1.1651 w_k1_FDX_s110
                                 + 0.918395 w_k1_IBM_s110
                                 + 1.04335 w_k1_INTC_s110
                                 + 0.919247 w_k1_KO_s110
                                 + 1.13628 w_k1_MAT_s110
                                 + 1.1331 w_k1_MSFT_s110 + 1.24799 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.918919 w_k1_ADM_s110
                                 + 1.19551 w_k1_AXP_s110
                                 + 0.848395 w_k1_DIS_s110
                                 + 0.93623 w_k1_FDX_s110
                                 + 1.04855 w_k1_IBM_s110
                                 + 1.06255 w_k1_INTC_s110
                                 + 0.929513 w_k1_KO_s110
                                 + 1.02592 w_k1_MAT_s110
                                 + 1.04586 w_k1_MSFT_s110
                                 + 1.07961 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.683255 w_k1_ADM_s110
                                 + 0.639651 w_k1_AXP_s110
                                 + 0.686844 w_k1_DIS_s110
                                 + 0.756368 w_k1_FDX_s110
                                 + 0.68535 w_k1_IBM_s110
                                 + 0.789145 w_k1_INTC_s110
                                 + 0.702086 w_k1_KO_s110
                                 + 0.689042 w_k1_MAT_s110
                                 + 0.747947 w_k1_MSFT_s110
                                 + 0.850635 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.967446 w_k1_ADM_s111
                                 + 1.10552 w_k1_AXP_s111
                                 + 1.09117 w_k1_DIS_s111
                                 + 1.05269 w_k1_FDX_s111
                                 + 0.883848 w_k1_IBM_s111
                                 + 0.839441 w_k1_INTC_s111
                                 + 0.926822 w_k1_KO_s111
                                 + 0.858964 w_k1_MAT_s111
                                 + 1.0333 w_k1_MSFT_s111 + 1.06764 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.753289 w_k1_ADM_s111
                                 + 0.65802 w_k1_AXP_s111
                                 + 0.631442 w_k1_DIS_s111
                                 + 0.730016 w_k1_FDX_s111
                                 + 0.708922 w_k1_IBM_s111
                                 + 0.87265 w_k1_INTC_s111
                                 + 0.730468 w_k1_KO_s111
                                 + 0.646205 w_k1_MAT_s111
                                 + 0.862545 w_k1_MSFT_s111
                                 + 0.834472 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.700509 w_k1_ADM_s111
                                 + 0.902805 w_k1_AXP_s111
                                 + 0.822412 w_k1_DIS_s111
                                 + 0.877323 w_k1_FDX_s111
                                 + 1.02616 w_k1_IBM_s111
                                 + 0.866344 w_k1_INTC_s111
                                 + 0.854951 w_k1_KO_s111
                                 + 1.23429 w_k1_MAT_s111
                                 + 0.981967 w_k1_MSFT_s111
                                 + 0.908991 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.877095 w_k1_ADM_s111
                                 + 1.03224 w_k1_AXP_s111
                                 + 1.41786 w_k1_DIS_s111
                                 + 1.00865 w_k1_FDX_s111
                                 + 1.61529 w_k1_IBM_s111
                                 + 0.95076 w_k1_INTC_s111
                                 + 1.41353 w_k1_KO_s111 + 1.91789 w_k1_MAT_s111
                                 + 1.17661 w_k1_MSFT_s111
                                 + 1.17239 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.80508 w_k1_ADM_s111
                                 + 0.851797 w_k1_AXP_s111
                                 + 0.913863 w_k1_DIS_s111
                                 + 0.813258 w_k1_FDX_s111
                                 + 1.26083 w_k1_IBM_s111
                                 + 0.803143 w_k1_INTC_s111
                                 + 0.917072 w_k1_KO_s111
                                 + 0.987031 w_k1_MAT_s111
                                 + 0.863373 w_k1_MSFT_s111
                                 + 0.90676 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.27485 w_k1_ADM_s111
                                 + 1.00357 w_k1_AXP_s111
                                 + 1.44675 w_k1_DIS_s111
                                 + 1.21961 w_k1_FDX_s111
                                 + 0.905018 w_k1_IBM_s111
                                 + 1.09967 w_k1_INTC_s111
                                 + 1.06762 w_k1_KO_s111
                                 + 0.996309 w_k1_MAT_s111
                                 + 1.17759 w_k1_MSFT_s111
                                 + 1.00071 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.90357 w_k1_ADM_s111
                                 + 1.18056 w_k1_AXP_s111
                                 + 1.30725 w_k1_DIS_s111
                                 + 1.14236 w_k1_FDX_s111
                                 + 1.07716 w_k1_IBM_s111
                                 + 1.04362 w_k1_INTC_s111
                                 + 1.05737 w_k1_KO_s111 + 1.48322 w_k1_MAT_s111
                                 + 1.112 w_k1_MSFT_s111 + 1.13079 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.37708 w_k1_ADM_s111
                                 + 1.51449 w_k1_AXP_s111
                                 + 1.68486 w_k1_DIS_s111
                                 + 1.26853 w_k1_FDX_s111
                                 + 1.30313 w_k1_IBM_s111
                                 + 1.06727 w_k1_INTC_s111
                                 + 1.19113 w_k1_KO_s111 + 1.54671 w_k1_MAT_s111
                                 + 1.29541 w_k1_MSFT_s111
                                 + 1.23713 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.910938 w_k1_ADM_s112
                                 + 1.1052 w_k1_AXP_s112 + 1.01879 w_k1_DIS_s112
                                 + 1.05645 w_k1_FDX_s112
                                 + 0.801658 w_k1_IBM_s112
                                 + 1.2149 w_k1_INTC_s112
                                 + 0.899777 w_k1_KO_s112
                                 + 0.815838 w_k1_MAT_s112
                                 + 1.00939 w_k1_MSFT_s112
                                 + 0.904429 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.96939 w_k1_ADM_s112
                                 + 0.933921 w_k1_AXP_s112
                                 + 0.993586 w_k1_DIS_s112
                                 + 0.991073 w_k1_FDX_s112
                                 + 1.24683 w_k1_IBM_s112
                                 + 0.990916 w_k1_INTC_s112
                                 + 1.12942 w_k1_KO_s112 + 1.23719 w_k1_MAT_s112
                                 + 0.99328 w_k1_MSFT_s112
                                 + 1.09777 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.649826 w_k1_ADM_s112
                                 + 0.814809 w_k1_AXP_s112
                                 + 0.728594 w_k1_DIS_s112
                                 + 0.757471 w_k1_FDX_s112
                                 + 0.882295 w_k1_IBM_s112
                                 + 0.862429 w_k1_INTC_s112
                                 + 0.765984 w_k1_KO_s112
                                 + 0.833826 w_k1_MAT_s112
                                 + 0.700852 w_k1_MSFT_s112
                                 + 0.77388 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.18355 w_k1_ADM_s112
                                 + 1.69182 w_k1_AXP_s112
                                 + 1.47384 w_k1_DIS_s112 + 1.2276 w_k1_FDX_s112
                                 + 1.7913 w_k1_IBM_s112
                                 + 1.52411 w_k1_INTC_s112
                                 + 1.54888 w_k1_KO_s112 + 1.82834 w_k1_MAT_s112
                                 + 1.25872 w_k1_MSFT_s112
                                 + 1.25971 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.80538 w_k1_ADM_s112
                                 + 1.05756 w_k1_AXP_s112
                                 + 1.56298 w_k1_DIS_s112
                                 + 1.21221 w_k1_FDX_s112
                                 + 0.959616 w_k1_IBM_s112
                                 + 1.33607 w_k1_INTC_s112
                                 + 1.14967 w_k1_KO_s112
                                 + 0.810603 w_k1_MAT_s112
                                 + 1.18956 w_k1_MSFT_s112
                                 + 1.16418 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.0358 w_k1_ADM_s112
                                 + 1.49099 w_k1_AXP_s112
                                 + 1.52584 w_k1_DIS_s112
                                 + 1.20487 w_k1_FDX_s112
                                 + 1.26837 w_k1_IBM_s112
                                 + 1.13325 w_k1_INTC_s112
                                 + 1.36524 w_k1_KO_s112 + 1.70253 w_k1_MAT_s112
                                 + 1.17917 w_k1_MSFT_s112
                                 + 1.18281 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.999909 w_k1_ADM_s112
                                 + 1.23581 w_k1_AXP_s112
                                 + 1.18484 w_k1_DIS_s112
                                 + 1.22279 w_k1_FDX_s112
                                 + 1.06627 w_k1_IBM_s112
                                 + 0.94817 w_k1_INTC_s112
                                 + 1.01333 w_k1_KO_s112 + 1.33064 w_k1_MAT_s112
                                 + 0.989103 w_k1_MSFT_s112
                                 + 1.07701 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.708651 w_k1_ADM_s112
                                 + 0.636044 w_k1_AXP_s112
                                 + 0.639765 w_k1_DIS_s112
                                 + 0.759767 w_k1_FDX_s112
                                 + 0.596408 w_k1_IBM_s112
                                 + 0.82332 w_k1_INTC_s112
                                 + 0.747313 w_k1_KO_s112
                                 + 0.56867 w_k1_MAT_s112
                                 + 0.751447 w_k1_MSFT_s112
                                 + 0.779077 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.12742 w_k1_ADM_s113
                                 + 1.63881 w_k1_AXP_s113
                                 + 1.05795 w_k1_DIS_s113
                                 + 1.01218 w_k1_FDX_s113
                                 + 1.56998 w_k1_IBM_s113
                                 + 1.30067 w_k1_INTC_s113
                                 + 1.30697 w_k1_KO_s113 + 1.97185 w_k1_MAT_s113
                                 + 1.0039 w_k1_MSFT_s113 + 1.11137 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.974909 w_k1_ADM_s113
                                 + 0.696363 w_k1_AXP_s113
                                 + 0.613814 w_k1_DIS_s113
                                 + 0.865166 w_k1_FDX_s113
                                 + 0.604511 w_k1_IBM_s113
                                 + 1.03528 w_k1_INTC_s113
                                 + 0.64758 w_k1_KO_s113 + 0.39858 w_k1_MAT_s113
                                 + 0.784198 w_k1_MSFT_s113
                                 + 0.781599 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.5719 w_k1_ADM_s113
                                 + 1.4203 w_k1_AXP_s113 + 1.84047 w_k1_DIS_s113
                                 + 1.40889 w_k1_FDX_s113
                                 + 1.50658 w_k1_IBM_s113
                                 + 1.45704 w_k1_INTC_s113
                                 + 1.62852 w_k1_KO_s113 + 1.86888 w_k1_MAT_s113
                                 + 1.41553 w_k1_MSFT_s113
                                 + 1.28593 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.01327 w_k1_ADM_s113
                                 + 0.814859 w_k1_AXP_s113
                                 + 0.668532 w_k1_DIS_s113
                                 + 0.835683 w_k1_FDX_s113
                                 + 0.785627 w_k1_IBM_s113
                                 + 0.856937 w_k1_INTC_s113
                                 + 0.746431 w_k1_KO_s113
                                 + 0.602041 w_k1_MAT_s113
                                 + 0.715446 w_k1_MSFT_s113
                                 + 0.811627 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15402 w_k1_ADM_s113
                                 + 1.23497 w_k1_AXP_s113
                                 + 1.18592 w_k1_DIS_s113 + 1.1508 w_k1_FDX_s113
                                 + 1.28379 w_k1_IBM_s113
                                 + 1.27984 w_k1_INTC_s113 + 1.2905 w_k1_KO_s113
                                 + 1.76887 w_k1_MAT_s113
                                 + 1.21314 w_k1_MSFT_s113
                                 + 1.20253 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.02527 w_k1_ADM_s113
                                 + 1.11334 w_k1_AXP_s113
                                 + 1.16895 w_k1_DIS_s113
                                 + 1.16869 w_k1_FDX_s113
                                 + 1.07229 w_k1_IBM_s113
                                 + 1.28676 w_k1_INTC_s113
                                 + 1.16067 w_k1_KO_s113 + 1.22933 w_k1_MAT_s113
                                 + 1.1949 w_k1_MSFT_s113 + 1.02013 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.89887 w_k1_ADM_s113
                                 + 0.922263 w_k1_AXP_s113
                                 + 1.1256 w_k1_DIS_s113 + 1.07956 w_k1_FDX_s113
                                 + 0.812817 w_k1_IBM_s113
                                 + 0.969851 w_k1_INTC_s113
                                 + 0.985429 w_k1_KO_s113
                                 + 0.929765 w_k1_MAT_s113
                                 + 1.09318 w_k1_MSFT_s113
                                 + 1.01126 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.849855 w_k1_ADM_s113
                                 + 0.833715 w_k1_AXP_s113
                                 + 0.838206 w_k1_DIS_s113
                                 + 0.860114 w_k1_FDX_s113
                                 + 1.14719 w_k1_IBM_s113
                                 + 0.942511 w_k1_INTC_s113
                                 + 1.06602 w_k1_KO_s113 + 1.06656 w_k1_MAT_s113
                                 + 1.00671 w_k1_MSFT_s113
                                 + 0.899554 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.07437 w_k1_ADM_s114
                                 + 0.783916 w_k1_AXP_s114
                                 + 0.636446 w_k1_DIS_s114
                                 + 0.751912 w_k1_FDX_s114
                                 + 0.925502 w_k1_IBM_s114
                                 + 0.84358 w_k1_INTC_s114
                                 + 0.855554 w_k1_KO_s114
                                 + 1.10129 w_k1_MAT_s114
                                 + 0.874895 w_k1_MSFT_s114
                                 + 0.901665 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.0253 w_k1_ADM_s114
                                 + 0.998807 w_k1_AXP_s114
                                 + 0.922063 w_k1_DIS_s114
                                 + 0.979682 w_k1_FDX_s114
                                 + 0.841924 w_k1_IBM_s114
                                 + 1.1336 w_k1_INTC_s114 + 1.08723 w_k1_KO_s114
                                 + 0.738132 w_k1_MAT_s114
                                 + 1.15539 w_k1_MSFT_s114
                                 + 0.966707 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.48691 w_k1_ADM_s114
                                 + 1.71522 w_k1_AXP_s114
                                 + 1.40473 w_k1_DIS_s114
                                 + 1.39953 w_k1_FDX_s114
                                 + 0.938048 w_k1_IBM_s114
                                 + 1.20875 w_k1_INTC_s114
                                 + 1.27382 w_k1_KO_s114 + 1.21655 w_k1_MAT_s114
                                 + 1.38244 w_k1_MSFT_s114
                                 + 1.19357 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.09153 w_k1_ADM_s114
                                 + 1.24867 w_k1_AXP_s114 + 1.3333 w_k1_DIS_s114
                                 + 1.1134 w_k1_FDX_s114 + 1.15796 w_k1_IBM_s114
                                 + 1.20297 w_k1_INTC_s114
                                 + 1.33356 w_k1_KO_s114 + 1.38152 w_k1_MAT_s114
                                 + 1.21268 w_k1_MSFT_s114
                                 + 1.28184 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.18454 w_k1_ADM_s114
                                 + 1.31654 w_k1_AXP_s114
                                 + 1.55969 w_k1_DIS_s114
                                 + 1.22417 w_k1_FDX_s114 + 1.6593 w_k1_IBM_s114
                                 + 1.43162 w_k1_INTC_s114
                                 + 1.81393 w_k1_KO_s114 + 2.57804 w_k1_MAT_s114
                                 + 1.40619 w_k1_MSFT_s114
                                 + 1.34178 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.01029 w_k1_ADM_s114
                                 + 0.783549 w_k1_AXP_s114
                                 + 0.965658 w_k1_DIS_s114
                                 + 0.82097 w_k1_FDX_s114 + 1.254 w_k1_IBM_s114
                                 + 0.884612 w_k1_INTC_s114
                                 + 1.20519 w_k1_KO_s114 + 1.24757 w_k1_MAT_s114
                                 + 0.890313 w_k1_MSFT_s114
                                 + 0.993301 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.93919 w_k1_ADM_s114
                                 + 0.646678 w_k1_AXP_s114
                                 + 0.606985 w_k1_DIS_s114
                                 + 0.700647 w_k1_FDX_s114
                                 + 0.627552 w_k1_IBM_s114
                                 + 0.7666 w_k1_INTC_s114
                                 + 0.758808 w_k1_KO_s114
                                 + 0.662841 w_k1_MAT_s114
                                 + 0.848305 w_k1_MSFT_s114
                                 + 0.819882 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.23284 w_k1_ADM_s114
                                 + 1.04859 w_k1_AXP_s114
                                 + 1.45642 w_k1_DIS_s114
                                 + 1.31156 w_k1_FDX_s114
                                 + 0.810536 w_k1_IBM_s114
                                 + 1.39015 w_k1_INTC_s114
                                 + 1.35273 w_k1_KO_s114
                                 + 0.911766 w_k1_MAT_s114
                                 + 1.36903 w_k1_MSFT_s114
                                 + 1.32376 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.04906 w_k1_ADM_s115
                                 + 0.968625 w_k1_AXP_s115 + 1.333 w_k1_DIS_s115
                                 + 1.00115 w_k1_FDX_s115 + 1.0163 w_k1_IBM_s115
                                 + 1.00322 w_k1_INTC_s115
                                 + 1.14641 w_k1_KO_s115
                                 + 0.701916 w_k1_MAT_s115
                                 + 1.08788 w_k1_MSFT_s115
                                 + 0.88398 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.00166 w_k1_ADM_s115
                                 + 1.11705 w_k1_AXP_s115
                                 + 1.25533 w_k1_DIS_s115
                                 + 1.20577 w_k1_FDX_s115
                                 + 0.764888 w_k1_IBM_s115
                                 + 1.12072 w_k1_INTC_s115
                                 + 0.857473 w_k1_KO_s115
                                 + 0.766283 w_k1_MAT_s115
                                 + 1.14604 w_k1_MSFT_s115
                                 + 0.954597 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.789501 w_k1_ADM_s115
                                 + 0.708909 w_k1_AXP_s115
                                 + 0.702082 w_k1_DIS_s115
                                 + 0.758141 w_k1_FDX_s115
                                 + 0.868225 w_k1_IBM_s115
                                 + 0.872885 w_k1_INTC_s115
                                 + 0.863539 w_k1_KO_s115
                                 + 0.579343 w_k1_MAT_s115
                                 + 0.797233 w_k1_MSFT_s115
                                 + 0.756713 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.03226 w_k1_ADM_s115
                                 + 0.884937 w_k1_AXP_s115
                                 + 0.899896 w_k1_DIS_s115
                                 + 0.950096 w_k1_FDX_s115
                                 + 0.745826 w_k1_IBM_s115
                                 + 0.936746 w_k1_INTC_s115
                                 + 0.850059 w_k1_KO_s115
                                 + 0.799053 w_k1_MAT_s115
                                 + 0.903851 w_k1_MSFT_s115
                                 + 0.784065 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.15601 w_k1_ADM_s115
                                 + 1.18508 w_k1_AXP_s115
                                 + 1.81074 w_k1_DIS_s115
                                 + 1.44095 w_k1_FDX_s115
                                 + 0.859517 w_k1_IBM_s115
                                 + 1.48843 w_k1_INTC_s115
                                 + 1.51434 w_k1_KO_s115 + 1.10653 w_k1_MAT_s115
                                 + 1.59798 w_k1_MSFT_s115
                                 + 1.16465 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.25141 w_k1_ADM_s115
                                 + 1.32115 w_k1_AXP_s115 + 1.3384 w_k1_DIS_s115
                                 + 1.16324 w_k1_FDX_s115
                                 + 1.08121 w_k1_IBM_s115
                                 + 1.07576 w_k1_INTC_s115
                                 + 1.22033 w_k1_KO_s115 + 1.01131 w_k1_MAT_s115
                                 + 1.11893 w_k1_MSFT_s115
                                 + 1.00165 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.17043 w_k1_ADM_s115
                                 + 1.07005 w_k1_AXP_s115
                                 + 0.897838 w_k1_DIS_s115
                                 + 0.893565 w_k1_FDX_s115
                                 + 1.27953 w_k1_IBM_s115
                                 + 0.869587 w_k1_INTC_s115
                                 + 1.10572 w_k1_KO_s115 + 1.25158 w_k1_MAT_s115
                                 + 0.857275 w_k1_MSFT_s115
                                 + 1.00713 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.49164 w_k1_ADM_s115
                                 + 1.4998 w_k1_AXP_s115 + 1.60631 w_k1_DIS_s115
                                 + 1.2359 w_k1_FDX_s115 + 1.58347 w_k1_IBM_s115
                                 + 1.23689 w_k1_INTC_s115
                                 + 1.65012 w_k1_KO_s115 + 1.85925 w_k1_MAT_s115
                                 + 1.38657 w_k1_MSFT_s115
                                 + 1.21062 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.780878 w_k1_ADM_s116
                                 + 0.896705 w_k1_AXP_s116
                                 + 1.14576 w_k1_DIS_s116
                                 + 1.05042 w_k1_FDX_s116
                                 + 1.13564 w_k1_IBM_s116
                                 + 1.17954 w_k1_INTC_s116
                                 + 1.16829 w_k1_KO_s116 + 1.31874 w_k1_MAT_s116
                                 + 1.08754 w_k1_MSFT_s116
                                 + 1.06886 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.976637 w_k1_ADM_s116
                                 + 1.32666 w_k1_AXP_s116
                                 + 1.37276 w_k1_DIS_s116
                                 + 1.21904 w_k1_FDX_s116 + 1.3786 w_k1_IBM_s116
                                 + 1.21257 w_k1_INTC_s116
                                 + 1.15399 w_k1_KO_s116 + 1.44759 w_k1_MAT_s116
                                 + 1.1418 w_k1_MSFT_s116 + 1.08825 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.913714 w_k1_ADM_s116
                                 + 1.03299 w_k1_AXP_s116
                                 + 0.818127 w_k1_DIS_s116
                                 + 1.06492 w_k1_FDX_s116
                                 + 0.848676 w_k1_IBM_s116
                                 + 1.04765 w_k1_INTC_s116
                                 + 0.838423 w_k1_KO_s116
                                 + 1.29924 w_k1_MAT_s116
                                 + 1.14895 w_k1_MSFT_s116
                                 + 0.969192 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.965926 w_k1_ADM_s116
                                 + 1.08137 w_k1_AXP_s116
                                 + 1.16028 w_k1_DIS_s116
                                 + 1.17671 w_k1_FDX_s116
                                 + 0.837311 w_k1_IBM_s116
                                 + 1.12856 w_k1_INTC_s116
                                 + 0.967622 w_k1_KO_s116
                                 + 0.732919 w_k1_MAT_s116
                                 + 1.04368 w_k1_MSFT_s116
                                 + 0.948104 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.677546 w_k1_ADM_s116
                                 + 0.628966 w_k1_AXP_s116
                                 + 0.868607 w_k1_DIS_s116
                                 + 0.854384 w_k1_FDX_s116
                                 + 0.785749 w_k1_IBM_s116
                                 + 0.948293 w_k1_INTC_s116
                                 + 0.958523 w_k1_KO_s116
                                 + 0.818271 w_k1_MAT_s116
                                 + 0.965859 w_k1_MSFT_s116
                                 + 0.861772 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.854895 w_k1_ADM_s116
                                 + 1.24538 w_k1_AXP_s116
                                 + 0.948791 w_k1_DIS_s116
                                 + 1.07897 w_k1_FDX_s116
                                 + 1.58496 w_k1_IBM_s116
                                 + 1.17244 w_k1_INTC_s116
                                 + 1.08407 w_k1_KO_s116 + 1.65339 w_k1_MAT_s116
                                 + 0.941159 w_k1_MSFT_s116
                                 + 1.01777 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.777437 w_k1_ADM_s116
                                 + 0.793294 w_k1_AXP_s116
                                 + 0.639603 w_k1_DIS_s116
                                 + 0.801407 w_k1_FDX_s116
                                 + 1.07419 w_k1_IBM_s116
                                 + 0.742436 w_k1_INTC_s116
                                 + 0.824245 w_k1_KO_s116
                                 + 1.11031 w_k1_MAT_s116
                                 + 0.852912 w_k1_MSFT_s116
                                 + 0.826119 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.18247 w_k1_ADM_s116
                                 + 1.42146 w_k1_AXP_s116
                                 + 1.76759 w_k1_DIS_s116
                                 + 1.47282 w_k1_FDX_s116
                                 + 1.31075 w_k1_IBM_s116
                                 + 1.34682 w_k1_INTC_s116
                                 + 1.62671 w_k1_KO_s116 + 2.13261 w_k1_MAT_s116
                                 + 1.47568 w_k1_MSFT_s116
                                 + 1.49299 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.799237 w_k1_ADM_s117
                                 + 0.656537 w_k1_AXP_s117
                                 + 0.599447 w_k1_DIS_s117
                                 + 0.69967 w_k1_FDX_s117
                                 + 0.818785 w_k1_IBM_s117
                                 + 0.671253 w_k1_INTC_s117
                                 + 0.721107 w_k1_KO_s117
                                 + 0.629625 w_k1_MAT_s117
                                 + 0.717289 w_k1_MSFT_s117
                                 + 0.851946 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.08753 w_k1_ADM_s117
                                 + 0.761101 w_k1_AXP_s117
                                 + 0.836091 w_k1_DIS_s117
                                 + 0.905556 w_k1_FDX_s117
                                 + 0.754275 w_k1_IBM_s117
                                 + 1.08008 w_k1_INTC_s117
                                 + 0.977847 w_k1_KO_s117
                                 + 0.644132 w_k1_MAT_s117
                                 + 0.942261 w_k1_MSFT_s117
                                 + 0.973034 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.895735 w_k1_ADM_s117
                                 + 1.22132 w_k1_AXP_s117
                                 + 1.20016 w_k1_DIS_s117 + 1.0948 w_k1_FDX_s117
                                 + 1.08978 w_k1_IBM_s117
                                 + 1.13811 w_k1_INTC_s117
                                 + 1.15065 w_k1_KO_s117 + 1.15198 w_k1_MAT_s117
                                 + 1.12047 w_k1_MSFT_s117
                                 + 1.09922 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15949 w_k1_ADM_s117
                                 + 1.41647 w_k1_AXP_s117
                                 + 1.26729 w_k1_DIS_s117 + 1.2237 w_k1_FDX_s117
                                 + 1.55894 w_k1_IBM_s117
                                 + 1.38247 w_k1_INTC_s117
                                 + 1.40324 w_k1_KO_s117 + 1.41891 w_k1_MAT_s117
                                 + 1.15264 w_k1_MSFT_s117
                                 + 1.33213 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.05839 w_k1_ADM_s117
                                 + 1.24399 w_k1_AXP_s117
                                 + 1.04923 w_k1_DIS_s117
                                 + 1.24724 w_k1_FDX_s117
                                 + 0.900524 w_k1_IBM_s117
                                 + 1.24886 w_k1_INTC_s117
                                 + 1.04065 w_k1_KO_s117 + 1.21703 w_k1_MAT_s117
                                 + 1.20518 w_k1_MSFT_s117 + 1.1579 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.824767 w_k1_ADM_s117
                                 + 0.867577 w_k1_AXP_s117
                                 + 0.842336 w_k1_DIS_s117
                                 + 0.833714 w_k1_FDX_s117
                                 + 1.06019 w_k1_IBM_s117
                                 + 0.989567 w_k1_INTC_s117
                                 + 0.983224 w_k1_KO_s117
                                 + 0.749008 w_k1_MAT_s117
                                 + 0.939096 w_k1_MSFT_s117
                                 + 0.857698 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.11767 w_k1_ADM_s117
                                 + 0.922491 w_k1_AXP_s117
                                 + 1.12185 w_k1_DIS_s117
                                 + 1.05273 w_k1_FDX_s117
                                 + 0.985955 w_k1_IBM_s117
                                 + 1.0274 w_k1_INTC_s117 + 1.06313 w_k1_KO_s117
                                 + 0.801145 w_k1_MAT_s117
                                 + 1.20788 w_k1_MSFT_s117 + 1.1327 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.861278 w_k1_ADM_s117
                                 + 1.01105 w_k1_AXP_s117
                                 + 0.930987 w_k1_DIS_s117
                                 + 0.942905 w_k1_FDX_s117
                                 + 1.29218 w_k1_IBM_s117
                                 + 1.11925 w_k1_INTC_s117
                                 + 1.08471 w_k1_KO_s117 + 1.11763 w_k1_MAT_s117
                                 + 0.929483 w_k1_MSFT_s117
                                 + 1.08356 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0244451338270563 w_k1_ADM_s000
                                 + 0.00995363407049414 w_k1_ADM_s001
                                 + 0.0202042276238162 w_k1_ADM_s002
                                 + 0.0315738262226944 w_k1_ADM_s003
                                 + 0.0129397940786799 w_k1_ADM_s004
                                 + 0.0148985981095897 w_k1_ADM_s005
                                 + 0.0437391306556572 w_k1_ADM_s006
                                 + 0.0475289545072443 w_k1_ADM_s007
                                 + 0.0192310712331976 w_k1_ADM_s010
                                 + 0.013135452496605 w_k1_ADM_s011
                                 + 0.0143377371421921 w_k1_ADM_s012
                                 + 0.0175938875021649 w_k1_ADM_s013
                                 + 0.044860348406826 w_k1_ADM_s014
                                 + 0.0436304674837977 w_k1_ADM_s015
                                 + 0.0287035769853143 w_k1_ADM_s016
                                 + 0.0238848389070289 w_k1_ADM_s017
                                 + 0.0279485988704224 w_k1_ADM_s100
                                 + 0.0208132546283836 w_k1_ADM_s101
                                 + 0.049344576465614 w_k1_ADM_s102
                                 + 0.078693553740552 w_k1_ADM_s103
                                 + 0.0972758710822468 w_k1_ADM_s104
                                 + 0.0499006082463296 w_k1_ADM_s105
                                 + 0.0476602497111949 w_k1_ADM_s106
                                 + 0.0478347539576534 w_k1_ADM_s107
                                 + 0.0217451837533104 w_k1_ADM_s110
                                 + 0.0316299501592812 w_k1_ADM_s111
                                 + 0.0179191936821813 w_k1_ADM_s112
                                 + 0.01995593937289 w_k1_ADM_s113
                                 + 0.0196097800165875 w_k1_ADM_s114
                                 + 0.033283063207388 w_k1_ADM_s115
                                 + 0.0267345524434099 w_k1_ADM_s116
                                 + 0.0121199826050934 w_k1_ADM_s117
                                 + 0.0270925662437805 w_k1_AXP_s000
                                 + 0.010265009519155 w_k1_AXP_s001
                                 + 0.0213123200535213 w_k1_AXP_s002
                                 + 0.0327806963264218 w_k1_AXP_s003
                                 + 0.0154490710396256 w_k1_AXP_s004
                                 + 0.0146754259659345 w_k1_AXP_s005
                                 + 0.0368143989473715 w_k1_AXP_s006
                                 + 0.0502394308136276 w_k1_AXP_s007
                                 + 0.0183619480791786 w_k1_AXP_s010
                                 + 0.0125120806872994 w_k1_AXP_s011
                                 + 0.013599919450999 w_k1_AXP_s012
                                 + 0.0180536294792851 w_k1_AXP_s013
                                 + 0.041526696267887 w_k1_AXP_s014
                                 + 0.0451828245795304 w_k1_AXP_s015
                                 + 0.0255371471029609 w_k1_AXP_s016
                                 + 0.0231545396666931 w_k1_AXP_s017
                                 + 0.0289556339055666 w_k1_AXP_s100
                                 + 0.0189679871019754 w_k1_AXP_s101
                                 + 0.0546135510386602 w_k1_AXP_s102
                                 + 0.0776509014819275 w_k1_AXP_s103
                                 + 0.107370945095801 w_k1_AXP_s104
                                 + 0.0556993629406956 w_k1_AXP_s105
                                 + 0.04066915468285 w_k1_AXP_s106
                                 + 0.0533661119971127 w_k1_AXP_s107
                                 + 0.0243299953591636 w_k1_AXP_s110
                                 + 0.0336852367269786 w_k1_AXP_s111
                                 + 0.0219131279899102 w_k1_AXP_s112
                                 + 0.019779897009781 w_k1_AXP_s113
                                 + 0.0185147048340577 w_k1_AXP_s114
                                 + 0.0316838775246586 w_k1_AXP_s115
                                 + 0.0304485320969997 w_k1_AXP_s116
                                 + 0.012520559532913 w_k1_AXP_s117
                                 + 0.0280889822482184 w_k1_DIS_s000
                                 + 0.00990228751466239 w_k1_DIS_s001
                                 + 0.0216450009533684 w_k1_DIS_s002
                                 + 0.0337121624834787 w_k1_DIS_s003
                                 + 0.0148344245766965 w_k1_DIS_s004
                                 + 0.0143003004284176 w_k1_DIS_s005
                                 + 0.0376718950224169 w_k1_DIS_s006
                                 + 0.0525437524100837 w_k1_DIS_s007
                                 + 0.0186871839825057 w_k1_DIS_s010
                                 + 0.0134020119176769 w_k1_DIS_s011
                                 + 0.0136535725189898 w_k1_DIS_s012
                                 + 0.0167837292690165 w_k1_DIS_s013
                                 + 0.0416161483655393 w_k1_DIS_s014
                                 + 0.046632113925294 w_k1_DIS_s015
                                 + 0.0272661012473914 w_k1_DIS_s016
                                 + 0.0253728040563963 w_k1_DIS_s017
                                 + 0.0307077253205521 w_k1_DIS_s100
                                 + 0.0194621237076449 w_k1_DIS_s101
                                 + 0.0496489692522976 w_k1_DIS_s102
                                 + 0.0844028962799089 w_k1_DIS_s103
                                 + 0.0968482897358421 w_k1_DIS_s104
                                 + 0.0558130738093506 w_k1_DIS_s105
                                 + 0.0452654402973766 w_k1_DIS_s106
                                 + 0.05034197759574 w_k1_DIS_s107
                                 + 0.0248646305109432 w_k1_DIS_s110
                                 + 0.0366431364349265 w_k1_DIS_s111
                                 + 0.0217606863447133 w_k1_DIS_s112
                                 + 0.0200154539325462 w_k1_DIS_s113
                                 + 0.0188200555966613 w_k1_DIS_s114
                                 + 0.0338604361604214 w_k1_DIS_s115
                                 + 0.0326496705747416 w_k1_DIS_s116
                                 + 0.0120781322025795 w_k1_DIS_s117
                                 + 0.0270333706690247 w_k1_FDX_s000
                                 + 0.00949727695230431 w_k1_FDX_s001
                                 + 0.0205324814323802 w_k1_FDX_s002
                                 + 0.0344171017529559 w_k1_FDX_s003
                                 + 0.0140553544112394 w_k1_FDX_s004
                                 + 0.0141531801717413 w_k1_FDX_s005
                                 + 0.0381196452730089 w_k1_FDX_s006
                                 + 0.0481238937453921 w_k1_FDX_s007
                                 + 0.0176370283319721 w_k1_FDX_s010
                                 + 0.0129928248150126 w_k1_FDX_s011
                                 + 0.01387129781235 w_k1_FDX_s012
                                 + 0.0171600317723388 w_k1_FDX_s013
                                 + 0.0397135271052323 w_k1_FDX_s014
                                 + 0.0477605388476339 w_k1_FDX_s015
                                 + 0.0273059849111216 w_k1_FDX_s016
                                 + 0.0230496697254597 w_k1_FDX_s017
                                 + 0.0281503081417986 w_k1_FDX_s100
                                 + 0.0204619542596883 w_k1_FDX_s101
                                 + 0.050819525622482 w_k1_FDX_s102
                                 + 0.0802251562889819 w_k1_FDX_s103
                                 + 0.0956460081738088 w_k1_FDX_s104
                                 + 0.0526766526144645 w_k1_FDX_s105
                                 + 0.0428629288834234 w_k1_FDX_s106
                                 + 0.0487023727382104 w_k1_FDX_s107
                                 + 0.0242863005779082 w_k1_FDX_s110
                                 + 0.0334799199010297 w_k1_FDX_s111
                                 + 0.0208077393403385 w_k1_FDX_s112
                                 + 0.0201208218587917 w_k1_FDX_s113
                                 + 0.0178367972415582 w_k1_FDX_s114
                                 + 0.0312965396090512 w_k1_FDX_s115
                                 + 0.0327439255194385 w_k1_FDX_s116
                                 + 0.0124137231405691 w_k1_FDX_s117
                                 + 0.026694734974919 w_k1_IBM_s000
                                 + 0.00971518673928966 w_k1_IBM_s001
                                 + 0.0204089304944911 w_k1_IBM_s002
                                 + 0.0342429500450783 w_k1_IBM_s003
                                 + 0.0143773062385215 w_k1_IBM_s004
                                 + 0.0142037622434508 w_k1_IBM_s005
                                 + 0.0370276318038884 w_k1_IBM_s006
                                 + 0.0482241503713351 w_k1_IBM_s007
                                 + 0.0188099991295097 w_k1_IBM_s010
                                 + 0.0117947326913665 w_k1_IBM_s011
                                 + 0.0131221964480497 w_k1_IBM_s012
                                 + 0.0168331973646586 w_k1_IBM_s013
                                 + 0.0418825851156949 w_k1_IBM_s014
                                 + 0.0459547289236492 w_k1_IBM_s015
                                 + 0.0267814314807025 w_k1_IBM_s016
                                 + 0.024723254618308 w_k1_IBM_s017
                                 + 0.0287231870850742 w_k1_IBM_s100
                                 + 0.0218736458519328 w_k1_IBM_s101
                                 + 0.0474339386375942 w_k1_IBM_s102
                                 + 0.0798220331196218 w_k1_IBM_s103
                                 + 0.0983145216502596 w_k1_IBM_s104
                                 + 0.0512991543110962 w_k1_IBM_s105
                                 + 0.0439261403215976 w_k1_IBM_s106
                                 + 0.0512511152688901 w_k1_IBM_s107
                                 + 0.0236781180485702 w_k1_IBM_s110
                                 + 0.0350125471776399 w_k1_IBM_s111
                                 + 0.0208827102604646 w_k1_IBM_s112
                                 + 0.0199514753476808 w_k1_IBM_s113
                                 + 0.0181573261956252 w_k1_IBM_s114
                                 + 0.0308105576913855 w_k1_IBM_s115
                                 + 0.0330860400736397 w_k1_IBM_s116
                                 + 0.0129564968485747 w_k1_IBM_s117
                                 + 0.0258917038415295 w_k1_INTC_s000
                                 + 0.0107976060323578 w_k1_INTC_s001
                                 + 0.0189265344482337 w_k1_INTC_s002
                                 + 0.0302439802653926 w_k1_INTC_s003
                                 + 0.01564008056366 w_k1_INTC_s004
                                 + 0.0143866536863698 w_k1_INTC_s005
                                 + 0.0403454757212759 w_k1_INTC_s006
                                 + 0.0508929961531806 w_k1_INTC_s007
                                 + 0.0197123600717218 w_k1_INTC_s010
                                 + 0.0136126018921496 w_k1_INTC_s011
                                 + 0.014199361128448 w_k1_INTC_s012
                                 + 0.0190005685805952 w_k1_INTC_s013
                                 + 0.0395751282071691 w_k1_INTC_s014
                                 + 0.0466787459780469 w_k1_INTC_s015
                                 + 0.0284816411852054 w_k1_INTC_s016
                                 + 0.0232705917980288 w_k1_INTC_s017
                                 + 0.0271058317394326 w_k1_INTC_s100
                                 + 0.0208440947857297 w_k1_INTC_s101
                                 + 0.0504905341791369 w_k1_INTC_s102
                                 + 0.081239555138483 w_k1_INTC_s103
                                 + 0.0940623834261909 w_k1_INTC_s104
                                 + 0.050025151776768 w_k1_INTC_s105
                                 + 0.0373739495906154 w_k1_INTC_s106
                                 + 0.0469266684952322 w_k1_INTC_s107
                                 + 0.0252537648521138 w_k1_INTC_s110
                                 + 0.0316119995358031 w_k1_INTC_s111
                                 + 0.021918556579967 w_k1_INTC_s112
                                 + 0.0214962125724886 w_k1_INTC_s113
                                 + 0.0194246712991509 w_k1_INTC_s114
                                 + 0.0316588841458055 w_k1_INTC_s115
                                 + 0.033443453311399 w_k1_INTC_s116
                                 + 0.0136306064645498 w_k1_INTC_s117
                                 + 0.0262611321013867 w_k1_KO_s000
                                 + 0.0102926723056299 w_k1_KO_s001
                                 + 0.0206012281446611 w_k1_KO_s002
                                 + 0.0308607688374632 w_k1_KO_s003
                                 + 0.0135438892487575 w_k1_KO_s004
                                 + 0.0161458754105489 w_k1_KO_s005
                                 + 0.0407493119198444 w_k1_KO_s006
                                 + 0.0514392337743469 w_k1_KO_s007
                                 + 0.0180599544779291 w_k1_KO_s010
                                 + 0.0122489379838607 w_k1_KO_s011
                                 + 0.0140736143180061 w_k1_KO_s012
                                 + 0.0173514852582401 w_k1_KO_s013
                                 + 0.0387374401928391 w_k1_KO_s014
                                 + 0.0467980656143892 w_k1_KO_s015
                                 + 0.0287262299277662 w_k1_KO_s016
                                 + 0.0268667620123511 w_k1_KO_s017
                                 + 0.027709598109864 w_k1_KO_s100
                                 + 0.0204121123891975 w_k1_KO_s101
                                 + 0.0443563781214146 w_k1_KO_s102
                                 + 0.0716507060124428 w_k1_KO_s103
                                 + 0.0926832571878855 w_k1_KO_s104
                                 + 0.0549663123540704 w_k1_KO_s105
                                 + 0.0455742903049867 w_k1_KO_s106
                                 + 0.0531695748474264 w_k1_KO_s107
                                 + 0.0226795358163914 w_k1_KO_s110
                                 + 0.0327943550328477 w_k1_KO_s111
                                 + 0.0206269925463578 w_k1_KO_s112
                                 + 0.0205498929831689 w_k1_KO_s113
                                 + 0.020993325508314 w_k1_KO_s114
                                 + 0.0334278501662645 w_k1_KO_s115
                                 + 0.032873068293925 w_k1_KO_s116
                                 + 0.0130831799073683 w_k1_KO_s117
                                 + 0.0330480279226109 w_k1_MAT_s000
                                 + 0.00846615777703543 w_k1_MAT_s001
                                 + 0.024124611611491 w_k1_MAT_s002
                                 + 0.0413704308831487 w_k1_MAT_s003
                                 + 0.0146563984897859 w_k1_MAT_s004
                                 + 0.0134959614901661 w_k1_MAT_s005
                                 + 0.0396620861326214 w_k1_MAT_s006
                                 + 0.0507020498085786 w_k1_MAT_s007
                                 + 0.0172100575122085 w_k1_MAT_s010
                                 + 0.0127000648039171 w_k1_MAT_s011
                                 + 0.012819544164694 w_k1_MAT_s012
                                 + 0.0158512728994675 w_k1_MAT_s013
                                 + 0.0377612378668032 w_k1_MAT_s014
                                 + 0.0431488094320785 w_k1_MAT_s015
                                 + 0.0291603480545658 w_k1_MAT_s016
                                 + 0.0262312126527252 w_k1_MAT_s017
                                 + 0.034684528737703 w_k1_MAT_s100
                                 + 0.028249084035361 w_k1_MAT_s101
                                 + 0.066373530469468 w_k1_MAT_s102
                                 + 0.0728735644342965 w_k1_MAT_s103
                                 + 0.0911497561572137 w_k1_MAT_s104
                                 + 0.0666493136183796 w_k1_MAT_s105
                                 + 0.0546849491558461 w_k1_MAT_s106
                                 + 0.0479691684239807 w_k1_MAT_s107
                                 + 0.0274970449028334 w_k1_MAT_s110
                                 + 0.0381825214925364 w_k1_MAT_s111
                                 + 0.0218135996204941 w_k1_MAT_s112
                                 + 0.0216390366189268 w_k1_MAT_s113
                                 + 0.0206014248790886 w_k1_MAT_s114
                                 + 0.029587174034755 w_k1_MAT_s115
                                 + 0.0380719553059286 w_k1_MAT_s116
                                 + 0.0119112751515152 w_k1_MAT_s117
                                 + 0.0276753562052589 w_k1_MSFT_s000
                                 + 0.0099428844861617 w_k1_MSFT_s001
                                 + 0.0199419941329538 w_k1_MSFT_s002
                                 + 0.033390075576745 w_k1_MSFT_s003
                                 + 0.0145401721015669 w_k1_MSFT_s004
                                 + 0.0146203460352797 w_k1_MSFT_s005
                                 + 0.0384019036675529 w_k1_MSFT_s006
                                 + 0.0477190653668029 w_k1_MSFT_s007
                                 + 0.0168734360388002 w_k1_MSFT_s010
                                 + 0.0125421824717486 w_k1_MSFT_s011
                                 + 0.0127946298021239 w_k1_MSFT_s012
                                 + 0.0167764160957323 w_k1_MSFT_s013
                                 + 0.036979516170842 w_k1_MSFT_s014
                                 + 0.047878373284625 w_k1_MSFT_s015
                                 + 0.029790328604389 w_k1_MSFT_s016
                                 + 0.0240012851338138 w_k1_MSFT_s017
                                 + 0.0259358276713942 w_k1_MSFT_s100
                                 + 0.0209339569945551 w_k1_MSFT_s101
                                 + 0.0491229002168335 w_k1_MSFT_s102
                                 + 0.080009995128312 w_k1_MSFT_s103
                                 + 0.094328918424969 w_k1_MSFT_s104
                                 + 0.0482522241351792 w_k1_MSFT_s105
                                 + 0.0411212674911897 w_k1_MSFT_s106
                                 + 0.0465116916757094 w_k1_MSFT_s107
                                 + 0.0250789020177025 w_k1_MSFT_s110
                                 + 0.0354443542009218 w_k1_MSFT_s111
                                 + 0.0196878199041715 w_k1_MSFT_s112
                                 + 0.0202512047399149 w_k1_MSFT_s113
                                 + 0.0199746695887525 w_k1_MSFT_s114
                                 + 0.0318966536008763 w_k1_MSFT_s115
                                 + 0.0332102981561989 w_k1_MSFT_s116
                                 + 0.0127244179422901 w_k1_MSFT_s117
                                 + 0.0268438642497007 w_k1_PG_s000
                                 + 0.00880582094366683 w_k1_PG_s001
                                 + 0.0205473930189657 w_k1_PG_s002
                                 + 0.0336160728427045 w_k1_PG_s003
                                 + 0.0137660694393597 w_k1_PG_s004
                                 + 0.0131765876143448 w_k1_PG_s005
                                 + 0.0394733395126754 w_k1_PG_s006
                                 + 0.04420779720779 w_k1_PG_s007
                                 + 0.0186337552047909 w_k1_PG_s010
                                 + 0.0116921640235886 w_k1_PG_s011
                                 + 0.0134731603983776 w_k1_PG_s012
                                 + 0.0157971974227471 w_k1_PG_s013
                                 + 0.0408125709264277 w_k1_PG_s014
                                 + 0.0410939062714162 w_k1_PG_s015
                                 + 0.0287354239587945 w_k1_PG_s016
                                 + 0.0245850996007519 w_k1_PG_s017
                                 + 0.0283000072496241 w_k1_PG_s100
                                 + 0.0187636684591478 w_k1_PG_s101
                                 + 0.0527557810506802 w_k1_PG_s102
                                 + 0.0792139312507708 w_k1_PG_s103
                                 + 0.0983626038876906 w_k1_PG_s104
                                 + 0.0545381520202556 w_k1_PG_s105
                                 + 0.0412165691848629 w_k1_PG_s106
                                 + 0.0518441882801389 w_k1_PG_s107
                                 + 0.025546976360875 w_k1_PG_s110
                                 + 0.0339866474587218 w_k1_PG_s111
                                 + 0.0200933425464141 w_k1_PG_s112
                                 + 0.0190713793927441 w_k1_PG_s113
                                 + 0.0192610089542141 w_k1_PG_s114
                                 + 0.0285507788379831 w_k1_PG_s115
                                 + 0.0313488842702505 w_k1_PG_s116
                                 + 0.0130660294629653 w_k1_PG_s117 >= 1.219
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
