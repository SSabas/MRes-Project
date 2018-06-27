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
 w_k1_ADM_s0_return_constraint:  1.10412 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.847549 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.877408 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.1067 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.849922 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.14246 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.53724 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.9989 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.887316 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.968113 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.07696 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.999302 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.24747 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.800038 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.894714 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.22934 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05036 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.805796 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.40209 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.09321 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.27939 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.07182 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.849946 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.09501 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.07523 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04323 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.895486 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.19114 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.898766 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.30751 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05232 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.41614 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.01004 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.738726 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.01425 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.02316 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.35589 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.16634 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20417 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.858489 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.938233 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.23536 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.725623 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.10807 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.58623 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.11847 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.723251 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.581309 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.873143 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.989958 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.17023 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.638155 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.717597 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.55249 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.928127 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.646782 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.13925 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.94025 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.35512 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.07843 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.766025 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.755744 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.16483 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.30529 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.02638 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.13386 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.863471 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.44904 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.870734 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.59831 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.875949 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.717282 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.17282 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.04122 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.3709 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.21726 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.24815 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.819058 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.868401 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.34853 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.716149 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.18045 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.78332 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.804274 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.760461 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.621333 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.0049 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.1856 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.23153 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.646546 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.703385 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.50392 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.924399 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.50023 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.33098 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.766227 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.56204 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.13865 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.646078 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.806685 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.51493 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.09148 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.758352 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.20754 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.96104 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.46078 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.80888 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.70876 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.994762 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.574678 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.22678 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.0697 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.24335 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.10216 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17327 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.870614 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.913525 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.14694 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.835916 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0989 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.47446 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.941505 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.813894 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.776524 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.07979 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.958026 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.174 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.711418 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.788351 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.32964 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.962748 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.706157 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.30181 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.778183 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.2287 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.10507 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.735122 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.942369 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.08994 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.01017 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.867941 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.29495 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.824056 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.32177 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.960619 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.45247 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.919386 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.727942 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.954983 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.03101 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.3724 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.04157 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13177 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.923544 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.839887 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.33236 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.752313 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.14166 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.30262 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.09846 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.11831 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.563547 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.704593 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.38267 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.23031 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.872817 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.884397 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.1226 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.861436 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.646206 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.75523 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.30393 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.91038 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.767772 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.30272 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.729748 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.67247 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.42291 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.957356 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.729125 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.26465 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.17118 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.798051 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.23858 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.25186 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.743061 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.83578 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.733131 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.708018 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.11557 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.06615 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.9758 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.968675 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.12311 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.811559 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1098 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.34819 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.925153 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.838276 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.709357 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.08949 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.13711 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.12895 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.804072 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.767875 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.18967 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.971504 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.882997 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.05446 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.778451 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.43293 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.06069 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.816631 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.845674 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.16081 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.01998 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.981682 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.03064 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.892729 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.23347 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.878908 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.07511 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.814745 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.753363 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.833059 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.935088 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.99781 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.935416 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13556 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.908782 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.893212 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.25052 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.757587 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.09925 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.30361 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.952756 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.936186 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.651885 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.854259 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.21509 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.27781 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.768505 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.803736 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.27484 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.875146 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.657886 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.06367 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.969175 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.59572 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.948398 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.97035 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.845591 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.42598 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.20686 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.908132 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.03556 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.06983 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.29976 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.829771 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.32757 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.1166 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.666835 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.32528 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.950967 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.9708 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.14525 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.09869 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.878043 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.747433 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.23667 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.654925 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.00471 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.8186 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.49642 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.18111 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.584108 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.775596 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.44839 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.50196 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.968314 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.895569 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.9423 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.1384 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.641943 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.859316 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.43403 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.42914 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.15935 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.06804 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.561334 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.28853 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.51806 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.03602 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.852148 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.68035 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.52296 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.775173 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.86134 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.37581 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.818785 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.94473 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.825538 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.14953 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.52598 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.07579 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.927892 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.990225 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.21028 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.788534 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03581 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.41814 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.885293 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.765709 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.78606 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.02753 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.01381 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08214 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.756872 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.751922 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.20667 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.95569 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.764027 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.24214 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.834737 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.34264 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0835 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.772106 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.941851 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.20976 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.00437 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.919109 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.27431 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.921375 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.3356 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.91511 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.25928 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.04168 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.813218 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.962112 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.03278 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.18957 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.04143 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13108 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.934743 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.871692 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.06456 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.82415 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.0418 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.334 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.987586 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.982858 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.813056 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.962561 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.10844 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.11897 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.85977 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.884659 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.13446 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.991957 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.803278 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.12295 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.925164 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.28704 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.14216 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.931398 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.952218 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.13605 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.13309 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00397 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.09928 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.978984 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.27615 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.946568 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.32655 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.997044 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.803625 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.15976 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.978549 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.07462 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.03742 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.354000046148754 z_k1_s0000
                                 - 0.615001581496472 z_k1_s0001
                                 - 0.132000126828404 z_k1_s0002
                                 - 0.282999822418246 z_k1_s0003
                                 - 0.396000099285071 z_k1_s0004
                                 - 0.258000005037213 z_k1_s0005
                                 - 0.347000974626506 z_k1_s0006
                                 - 0.426999602885416 z_k1_s0007
                                 - 0.13200017527522 z_k1_s0010
                                 - 0.160000091166817 z_k1_s0011
                                 - 0.246000290262876 z_k1_s0012
                                 - 0.203000590965231 z_k1_s0013
                                 - 0.228000229975311 z_k1_s0014
                                 - 0.190000458479682 z_k1_s0015
                                 - 0.107000055964679 z_k1_s0016
                                 - 0.334999965739682 z_k1_s0017
                                 - 0.860002300001548 z_k1_s0020
                                 - 0.925001383370984 z_k1_s0021
                                 - 0.44200126933125 z_k1_s0022
                                 - 0.208000349090545 z_k1_s0023
                                 - 0.441001103086646 z_k1_s0024
                                 - 0.423000924090932 z_k1_s0025
                                 - 0.980003493306309 z_k1_s0026
                                 - 0.410001107417045 z_k1_s0027
                                 - 0.315999299186325 z_k1_s0030
                                 - 0.106000096530923 z_k1_s0031
                                 - 0.329000746827202 z_k1_s0032
                                 - 0.394000500030064 z_k1_s0033
                                 - 0.37900055698325 z_k1_s0034
                                 - 0.301999851342733 z_k1_s0035
                                 - 0.195000008108739 z_k1_s0036
                                 - 0.474000196279741 z_k1_s0037
                                 - 0.744998149443254 z_k1_s0040
                                 - 1.63000274140867 z_k1_s0041
                                 - 0.437000363417028 z_k1_s0042
                                 - 0.694000891817178 z_k1_s0043
                                 - 1.81300149304934 z_k1_s0044
                                 - 0.626000510810192 z_k1_s0045
                                 - 0.550000881303479 z_k1_s0046
                                 - 0.630001203811525 z_k1_s0047
                                 - 0.469002810609166 z_k1_s0050
                                 - 0.542003458368236 z_k1_s0051
                                 - 0.724002709042415 z_k1_s0052
                                 - 0.273001341624635 z_k1_s0053
                                 - 0.26800138591833 z_k1_s0054
                                 - 0.475003497140742 z_k1_s0055
                                 - 0.17500091533404 z_k1_s0056
                                 - 0.156000898729079 z_k1_s0057
                                 - 0.580001458672559 z_k1_s0060
                                 - 0.483001013801062 z_k1_s0061
                                 - 0.996002176112917 z_k1_s0062
                                 - 0.484001089378228 z_k1_s0063
                                 - 0.787999168887502 z_k1_s0064
                                 - 0.797004086690375 z_k1_s0065
                                 - 0.529000782443372 z_k1_s0066
                                 - 0.640000696292053 z_k1_s0067
                                 - 0.413001363811493 z_k1_s0070
                                 - 0.172000022121545 z_k1_s0071
                                 - 0.296000370944318 z_k1_s0072
                                 - 0.282001147832717 z_k1_s0073
                                 - 0.270000718307618 z_k1_s0074
                                 - 0.364000248917848 z_k1_s0075
                                 - 0.473000955434958 z_k1_s0076
                                 - 0.28600129100775 z_k1_s0077
                                 - 0.525999656224306 z_k1_s0100
                                 - 0.27900023429063 z_k1_s0101
                                 - 0.145000302425477 z_k1_s0102
                                 - 0.271001248703003 z_k1_s0103
                                 - 0.302001274735295 z_k1_s0104
                                 - 0.426001378864464 z_k1_s0105
                                 - 0.234000400727896 z_k1_s0106
                                 - 0.252000918553763 z_k1_s0107
                                 - 0.152000232028813 z_k1_s0110
                                 - 0.262000130450823 z_k1_s0111
                                 - 0.322999744698292 z_k1_s0112
                                 - 0.255999992764085 z_k1_s0113
                                 - 0.69800047261486 z_k1_s0114
                                 - 0.225000121649758 z_k1_s0115
                                 - 0.676999045410727 z_k1_s0116
                                 - 0.558001461260074 z_k1_s0117
                                 - 0.410999948261101 z_k1_s0120
                                 - 0.520000833996534 z_k1_s0121
                                 - 0.396999759618975 z_k1_s0122
                                 - 0.940001692262866 z_k1_s0123
                                 - 0.561999959623678 z_k1_s0124
                                 - 0.378999860007487 z_k1_s0125
                                 - 0.758000680214463 z_k1_s0126
                                 - 0.834000469729518 z_k1_s0127
                                 - 0.118999278675995 z_k1_s0130
                                 - 0.0489999564602856 z_k1_s0131
                                 - 0.0449999621594424 z_k1_s0132
                                 - 0.1259999781263 z_k1_s0133
                                 - 0.181999267739145 z_k1_s0134
                                 - 0.186999917489115 z_k1_s0135
                                 - 0.155000173282023 z_k1_s0136
                                 - 0.18799941684015 z_k1_s0137
                                 - 0.204999933562355 z_k1_s0140
                                 - 0.197999440426424 z_k1_s0141
                                 - 0.205999753867662 z_k1_s0142
                                 - 0.202000472646434 z_k1_s0143
                                 - 0.122999960137413 z_k1_s0144
                                 - 0.217999348530127 z_k1_s0145
                                 - 0.159999965229143 z_k1_s0146
                                 - 0.438999908974958 z_k1_s0147
                                 - 0.192999734815445 z_k1_s0150
                                 - 0.6630011282841 z_k1_s0151
                                 - 0.643998540364795 z_k1_s0152
                                 - 0.689001470491161 z_k1_s0153
                                 - 0.470999770139025 z_k1_s0154
                                 - 0.358999990953861 z_k1_s0155
                                 - 0.275999862884653 z_k1_s0156
                                 - 0.503997866633843 z_k1_s0157
                                 - 0.171999794060819 z_k1_s0160
                                 - 0.19400048171962 z_k1_s0161
                                 - 0.143000255783038 z_k1_s0162
                                 - 0.132000513953519 z_k1_s0163
                                 - 0.0580000397553708 z_k1_s0164
                                 - 0.0809999268170087 z_k1_s0165
                                 - 0.155999841217946 z_k1_s0166
                                 - 0.267999511118053 z_k1_s0167
                                 - 0.154999815266815 z_k1_s0170
                                 - 0.160999850648713 z_k1_s0171
                                 - 0.27500032900496 z_k1_s0172
                                 - 0.349998832280684 z_k1_s0173
                                 - 0.264000393404681 z_k1_s0174
                                 - 0.195999733876781 z_k1_s0175
                                 - 0.338998896680405 z_k1_s0176
                                 - 0.198999363768131 z_k1_s0177
                                 - 0.104999872838289 z_k1_s1000
                                 - 0.24199955451318 z_k1_s1001
                                 - 0.134000353564471 z_k1_s1002
                                 - 0.0759998845115497 z_k1_s1003
                                 - 0.143000184393646 z_k1_s1004
                                 - 0.292000256006742 z_k1_s1005
                                 - 0.103999932668334 z_k1_s1006
                                 - 0.134999924434843 z_k1_s1007
                                 - 0.294999862594101 z_k1_s1010
                                 - 0.748997865721899 z_k1_s1011
                                 - 0.606000344489127 z_k1_s1012
                                 - 0.247999681571458 z_k1_s1013
                                 - 0.361999223844045 z_k1_s1014
                                 - 0.538001077357779 z_k1_s1015
                                 - 0.350999904948864 z_k1_s1016
                                 - 0.392998785565764 z_k1_s1017
                                 - 0.0800000390760988 z_k1_s1020
                                 - 0.20499989501375 z_k1_s1021
                                 - 0.0790000352613975 z_k1_s1022
                                 - 0.133000329955268 z_k1_s1023
                                 - 0.106999698573027 z_k1_s1024
                                 - 0.0429999866321526 z_k1_s1025
                                 - 0.128999959896458 z_k1_s1026
                                 - 0.111000068631837 z_k1_s1027
                                 - 0.19899967023675 z_k1_s1030
                                 - 0.054999939399586 z_k1_s1031
                                 - 0.0449999334178533 z_k1_s1032
                                 - 0.113000291993444 z_k1_s1033
                                 - 0.0940001684282196 z_k1_s1034
                                 - 0.159999908707835 z_k1_s1035
                                 - 0.188000336856439 z_k1_s1036
                                 - 0.0809999362521022 z_k1_s1037
                                 - 0.276000422005896 z_k1_s1040
                                 - 0.84699894310476 z_k1_s1041
                                 - 0.237999589839514 z_k1_s1042
                                 - 0.390999913092657 z_k1_s1043
                                 - 0.370000029841997 z_k1_s1044
                                 - 0.196999830821542 z_k1_s1045
                                 - 0.231999544939437 z_k1_s1046
                                 - 0.187999763471425 z_k1_s1047
                                 - 0.175001068746754 z_k1_s1050
                                 - 0.284000790397141 z_k1_s1051
                                 - 0.152000920214295 z_k1_s1052
                                 - 0.0710002681995325 z_k1_s1053
                                 - 0.231001184824924 z_k1_s1054
                                 - 0.215000344799765 z_k1_s1055
                                 - 0.0770003008079785 z_k1_s1056
                                 - 0.207001336792129 z_k1_s1057
                                 - 0.0709999369455898 z_k1_s1060
                                 - 0.151000228760197 z_k1_s1061
                                 - 0.0419999747227385 z_k1_s1062
                                 - 0.0419999747227385 z_k1_s1063
                                 - 0.155999994798708 z_k1_s1064
                                 - 0.0580000020456721 z_k1_s1065
                                 - 0.0899996686916913 z_k1_s1066
                                 - 0.166000302875428 z_k1_s1067
                                 - 0.183999817124653 z_k1_s1070
                                 - 0.101999959725604 z_k1_s1071
                                 - 0.0990000068366018 z_k1_s1072
                                 - 0.0559999533944567 z_k1_s1073
                                 - 0.112999474685374 z_k1_s1074
                                 - 0.208000064836481 z_k1_s1075
                                 - 0.101999959725604 z_k1_s1076
                                 - 0.177000135250251 z_k1_s1077
                                 - 0.566001789283992 z_k1_s1100
                                 - 0.596001115972454 z_k1_s1101
                                 - 0.455000936130142 z_k1_s1102
                                 - 0.93300024125259 z_k1_s1103
                                 - 0.601998751705784 z_k1_s1104
                                 - 0.787000173628868 z_k1_s1105
                                 - 0.586999088361553 z_k1_s1106
                                 - 1.04800323423593 z_k1_s1107
                                 - 0.893999346986455 z_k1_s1110
                                 - 0.362999940321282 z_k1_s1111
                                 - 0.534998466226029 z_k1_s1112
                                 - 0.516999119803285 z_k1_s1113
                                 - 0.36599983139174 z_k1_s1114
                                 - 0.54200020005648 z_k1_s1115
                                 - 0.489000136479015 z_k1_s1116
                                 - 0.269999738738211 z_k1_s1117
                                 - 0.509998228727049 z_k1_s1120
                                 - 0.93299632225442 z_k1_s1121
                                 - 0.571998823084986 z_k1_s1122
                                 - 0.446998444384894 z_k1_s1123
                                 - 0.338998737670017 z_k1_s1124
                                 - 1.04999462270832 z_k1_s1125
                                 - 1.10799899202766 z_k1_s1126
                                 - 0.38999861403255 z_k1_s1127
                                 - 0.671998946722835 z_k1_s1130
                                 - 1.62500039339847 z_k1_s1131
                                 - 0.732002297980602 z_k1_s1132
                                 - 0.536999722635134 z_k1_s1133
                                 - 0.943999946037897 z_k1_s1134
                                 - 0.736000747266101 z_k1_s1135
                                 - 1.02400219671698 z_k1_s1136
                                 - 0.382999548199037 z_k1_s1137
                                 - 0.752999769528268 z_k1_s1140
                                 - 0.4859977991831 z_k1_s1141
                                 - 0.508996922623874 z_k1_s1142
                                 - 0.227999490238821 z_k1_s1143
                                 - 0.805999457669827 z_k1_s1144
                                 - 0.471999281651393 z_k1_s1145
                                 - 0.441998716950608 z_k1_s1146
                                 - 0.668999934361396 z_k1_s1147
                                 - 0.39999919597688 z_k1_s1150
                                 - 1.28299692612046 z_k1_s1151
                                 - 0.352999339687985 z_k1_s1152
                                 - 0.421998908155158 z_k1_s1153
                                 - 1.09599989090393 z_k1_s1154
                                 - 0.45299887953279 z_k1_s1155
                                 - 0.650998880631445 z_k1_s1156
                                 - 0.525000434828793 z_k1_s1157
                                 - 0.66199919828236 z_k1_s1160
                                 - 0.196000032065371 z_k1_s1161
                                 - 0.287999856161433 z_k1_s1162
                                 - 0.386000018394063 z_k1_s1163
                                 - 0.187999971083133 z_k1_s1164
                                 - 0.36299984307013 z_k1_s1165
                                 - 0.510000963618744 z_k1_s1166
                                 - 0.33099959914118 z_k1_s1167
                                 - 0.430001223819761 z_k1_s1170
                                 - 0.383999963248085 z_k1_s1171
                                 - 0.707998912707173 z_k1_s1172
                                 - 0.577999132831566 z_k1_s1173
                                 - 0.698001160409562 z_k1_s1174
                                 - 0.185000105015382 z_k1_s1175
                                 - 0.298000080984216 z_k1_s1176
                                 - 0.344000254055642 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.899334 w_k1_ADM_s000
                                 + 0.88909 w_k1_AXP_s000
                                 + 1.19457 w_k1_DIS_s000
                                 + 0.964866 w_k1_FDX_s000
                                 + 1.19205 w_k1_IBM_s000
                                 + 1.01518 w_k1_INTC_s000
                                 + 1.29975 w_k1_KO_s000
                                 + 0.948138 w_k1_MAT_s000
                                 + 1.10789 w_k1_MSFT_s000
                                 + 1.09019 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.06375 w_k1_ADM_s000
                                 + 0.976774 w_k1_AXP_s000
                                 + 1.00553 w_k1_DIS_s000
                                 + 1.07065 w_k1_FDX_s000
                                 + 0.752634 w_k1_IBM_s000
                                 + 0.871119 w_k1_INTC_s000
                                 + 0.990734 w_k1_KO_s000
                                 + 0.835327 w_k1_MAT_s000
                                 + 0.924957 w_k1_MSFT_s000
                                 + 0.974902 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.20019 w_k1_ADM_s000
                                 + 1.42148 w_k1_AXP_s000
                                 + 1.97888 w_k1_DIS_s000
                                 + 1.59599 w_k1_FDX_s000
                                 + 1.12771 w_k1_IBM_s000
                                 + 1.19805 w_k1_INTC_s000
                                 + 1.43499 w_k1_KO_s000 + 1.28885 w_k1_MAT_s000
                                 + 1.13042 w_k1_MSFT_s000
                                 + 1.21233 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.924342 w_k1_ADM_s000
                                 + 0.971559 w_k1_AXP_s000
                                 + 1.21517 w_k1_DIS_s000
                                 + 0.980594 w_k1_FDX_s000
                                 + 1.72242 w_k1_IBM_s000
                                 + 0.978262 w_k1_INTC_s000
                                 + 1.43432 w_k1_KO_s000 + 1.7208 w_k1_MAT_s000
                                 + 1.04629 w_k1_MSFT_s000
                                 + 0.966677 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.77609 w_k1_ADM_s000
                                 + 0.835083 w_k1_AXP_s000
                                 + 0.76035 w_k1_DIS_s000
                                 + 0.738257 w_k1_FDX_s000
                                 + 1.22223 w_k1_IBM_s000
                                 + 0.740928 w_k1_INTC_s000
                                 + 0.978334 w_k1_KO_s000
                                 + 1.03805 w_k1_MAT_s000
                                 + 0.712083 w_k1_MSFT_s000
                                 + 0.930876 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.805154 w_k1_ADM_s000
                                 + 0.773772 w_k1_AXP_s000
                                 + 0.683139 w_k1_DIS_s000
                                 + 0.88983 w_k1_FDX_s000
                                 + 0.603904 w_k1_IBM_s000
                                 + 0.959982 w_k1_INTC_s000
                                 + 0.780368 w_k1_KO_s000
                                 + 0.619916 w_k1_MAT_s000
                                 + 0.807993 w_k1_MSFT_s000
                                 + 0.927802 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.45302 w_k1_ADM_s000
                                 + 1.26855 w_k1_AXP_s000
                                 + 1.42945 w_k1_DIS_s000
                                 + 1.23454 w_k1_FDX_s000
                                 + 0.994797 w_k1_IBM_s000
                                 + 1.19726 w_k1_INTC_s000
                                 + 1.18457 w_k1_KO_s000
                                 + 0.774473 w_k1_MAT_s000
                                 + 1.1521 w_k1_MSFT_s000 + 1.12177 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.930047 w_k1_ADM_s000
                                 + 0.553112 w_k1_AXP_s000
                                 + 0.727569 w_k1_DIS_s000
                                 + 0.781149 w_k1_FDX_s000
                                 + 0.638189 w_k1_IBM_s000
                                 + 0.896674 w_k1_INTC_s000
                                 + 0.772659 w_k1_KO_s000
                                 + 0.575436 w_k1_MAT_s000
                                 + 0.855397 w_k1_MSFT_s000
                                 + 0.838298 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.774965 w_k1_ADM_s001
                                 + 0.717432 w_k1_AXP_s001
                                 + 0.711108 w_k1_DIS_s001
                                 + 0.750391 w_k1_FDX_s001
                                 + 0.988552 w_k1_IBM_s001
                                 + 0.735616 w_k1_INTC_s001
                                 + 0.8151 w_k1_KO_s001 + 0.856918 w_k1_MAT_s001
                                 + 0.736317 w_k1_MSFT_s001
                                 + 0.948541 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.909608 w_k1_ADM_s001
                                 + 0.914265 w_k1_AXP_s001
                                 + 1.13156 w_k1_DIS_s001
                                 + 0.858608 w_k1_FDX_s001
                                 + 1.2763 w_k1_IBM_s001
                                 + 1.04127 w_k1_INTC_s001
                                 + 1.14956 w_k1_KO_s001 + 1.16154 w_k1_MAT_s001
                                 + 1.107 w_k1_MSFT_s001 + 1.0156 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.09189 w_k1_ADM_s001
                                 + 1.00304 w_k1_AXP_s001
                                 + 1.07629 w_k1_DIS_s001
                                 + 1.03525 w_k1_FDX_s001
                                 + 0.930192 w_k1_IBM_s001
                                 + 0.921997 w_k1_INTC_s001
                                 + 0.938976 w_k1_KO_s001 + 1.0491 w_k1_MAT_s001
                                 + 0.873252 w_k1_MSFT_s001
                                 + 1.08471 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.08322 w_k1_ADM_s001
                                 + 1.17194 w_k1_AXP_s001
                                 + 1.58614 w_k1_DIS_s001
                                 + 1.28165 w_k1_FDX_s001
                                 + 0.853165 w_k1_IBM_s001
                                 + 1.17471 w_k1_INTC_s001
                                 + 1.09966 w_k1_KO_s001
                                 + 0.792554 w_k1_MAT_s001
                                 + 1.19741 w_k1_MSFT_s001
                                 + 1.04257 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.00181 w_k1_ADM_s001
                                 + 0.976765 w_k1_AXP_s001
                                 + 0.936988 w_k1_DIS_s001
                                 + 0.992625 w_k1_FDX_s001
                                 + 0.687024 w_k1_IBM_s001
                                 + 0.885881 w_k1_INTC_s001
                                 + 0.851323 w_k1_KO_s001
                                 + 0.684149 w_k1_MAT_s001
                                 + 0.92583 w_k1_MSFT_s001 + 1.0001 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.816744 w_k1_ADM_s001
                                 + 0.900501 w_k1_AXP_s001
                                 + 0.915565 w_k1_DIS_s001
                                 + 0.885781 w_k1_FDX_s001
                                 + 0.94222 w_k1_IBM_s001
                                 + 1.01984 w_k1_INTC_s001
                                 + 0.953875 w_k1_KO_s001
                                 + 0.851134 w_k1_MAT_s001
                                 + 0.995841 w_k1_MSFT_s001
                                 + 0.988282 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.855045 w_k1_ADM_s001
                                 + 0.73518 w_k1_AXP_s001
                                 + 0.650562 w_k1_DIS_s001
                                 + 0.723286 w_k1_FDX_s001
                                 + 0.705526 w_k1_IBM_s001
                                 + 0.817749 w_k1_INTC_s001
                                 + 0.723335 w_k1_KO_s001
                                 + 0.693805 w_k1_MAT_s001
                                 + 0.690936 w_k1_MSFT_s001
                                 + 0.753764 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.05845 w_k1_ADM_s001
                                 + 1.20103 w_k1_AXP_s001
                                 + 1.45668 w_k1_DIS_s001
                                 + 1.13844 w_k1_FDX_s001
                                 + 1.38624 w_k1_IBM_s001
                                 + 1.30239 w_k1_INTC_s001
                                 + 1.36089 w_k1_KO_s001 + 1.33654 w_k1_MAT_s001
                                 + 1.12416 w_k1_MSFT_s001
                                 + 1.03541 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.22176 w_k1_ADM_s002
                                 + 1.40758 w_k1_AXP_s002
                                 + 1.48174 w_k1_DIS_s002
                                 + 1.26816 w_k1_FDX_s002
                                 + 0.942513 w_k1_IBM_s002
                                 + 1.34616 w_k1_INTC_s002 + 1.2075 w_k1_KO_s002
                                 + 0.872031 w_k1_MAT_s002
                                 + 1.3543 w_k1_MSFT_s002 + 1.0803 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.09706 w_k1_ADM_s002
                                 + 0.888103 w_k1_AXP_s002
                                 + 0.945984 w_k1_DIS_s002
                                 + 0.954281 w_k1_FDX_s002
                                 + 0.697561 w_k1_IBM_s002
                                 + 1.19688 w_k1_INTC_s002 + 0.7856 w_k1_KO_s002
                                 + 0.664788 w_k1_MAT_s002
                                 + 1.00276 w_k1_MSFT_s002
                                 + 0.90127 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.981879 w_k1_ADM_s002
                                 + 1.30357 w_k1_AXP_s002
                                 + 1.20614 w_k1_DIS_s002
                                 + 0.948498 w_k1_FDX_s002
                                 + 1.65273 w_k1_IBM_s002
                                 + 1.21525 w_k1_INTC_s002
                                 + 1.31404 w_k1_KO_s002 + 1.85943 w_k1_MAT_s002
                                 + 1.1704 w_k1_MSFT_s002 + 1.14447 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.810407 w_k1_ADM_s002
                                 + 0.656721 w_k1_AXP_s002
                                 + 0.556056 w_k1_DIS_s002
                                 + 0.735972 w_k1_FDX_s002
                                 + 0.633719 w_k1_IBM_s002
                                 + 0.866845 w_k1_INTC_s002
                                 + 0.583204 w_k1_KO_s002
                                 + 0.584092 w_k1_MAT_s002
                                 + 0.833036 w_k1_MSFT_s002
                                 + 0.749586 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.885661 w_k1_ADM_s002
                                 + 0.772366 w_k1_AXP_s002
                                 + 0.607513 w_k1_DIS_s002
                                 + 0.770467 w_k1_FDX_s002
                                 + 0.926724 w_k1_IBM_s002
                                 + 0.91698 w_k1_INTC_s002
                                 + 0.848449 w_k1_KO_s002
                                 + 0.86126 w_k1_MAT_s002
                                 + 0.927979 w_k1_MSFT_s002
                                 + 0.808076 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.819775 w_k1_ADM_s002
                                 + 0.923025 w_k1_AXP_s002
                                 + 0.828452 w_k1_DIS_s002
                                 + 0.790802 w_k1_FDX_s002
                                 + 1.21625 w_k1_IBM_s002
                                 + 0.895795 w_k1_INTC_s002
                                 + 1.07249 w_k1_KO_s002 + 1.3333 w_k1_MAT_s002
                                 + 0.956158 w_k1_MSFT_s002
                                 + 0.864158 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.944337 w_k1_ADM_s002
                                 + 1.02814 w_k1_AXP_s002 + 1.1066 w_k1_DIS_s002
                                 + 0.944893 w_k1_FDX_s002
                                 + 1.03614 w_k1_IBM_s002
                                 + 1.08188 w_k1_INTC_s002
                                 + 1.02887 w_k1_KO_s002 + 1.09008 w_k1_MAT_s002
                                 + 1.23923 w_k1_MSFT_s002
                                 + 0.958328 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.31839 w_k1_ADM_s002
                                 + 1.30776 w_k1_AXP_s002
                                 + 1.31742 w_k1_DIS_s002
                                 + 1.12348 w_k1_FDX_s002
                                 + 1.37981 w_k1_IBM_s002
                                 + 1.26611 w_k1_INTC_s002
                                 + 1.41787 w_k1_KO_s002 + 1.33529 w_k1_MAT_s002
                                 + 1.30416 w_k1_MSFT_s002
                                 + 1.05254 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.832208 w_k1_ADM_s003
                                 + 0.772228 w_k1_AXP_s003
                                 + 0.731009 w_k1_DIS_s003
                                 + 0.69329 w_k1_FDX_s003
                                 + 1.02109 w_k1_IBM_s003
                                 + 0.950782 w_k1_INTC_s003
                                 + 0.964618 w_k1_KO_s003
                                 + 0.873656 w_k1_MAT_s003
                                 + 0.770851 w_k1_MSFT_s003
                                 + 0.827185 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.892423 w_k1_ADM_s003
                                 + 0.906085 w_k1_AXP_s003
                                 + 1.00026 w_k1_DIS_s003
                                 + 0.972603 w_k1_FDX_s003
                                 + 0.820904 w_k1_IBM_s003
                                 + 1.27977 w_k1_INTC_s003
                                 + 0.916557 w_k1_KO_s003
                                 + 0.837637 w_k1_MAT_s003
                                 + 1.03971 w_k1_MSFT_s003
                                 + 1.03531 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.929653 w_k1_ADM_s003
                                 + 1.1565 w_k1_AXP_s003
                                 + 0.833871 w_k1_DIS_s003
                                 + 1.05306 w_k1_FDX_s003
                                 + 0.639711 w_k1_IBM_s003
                                 + 0.993558 w_k1_INTC_s003
                                 + 0.652536 w_k1_KO_s003
                                 + 0.831059 w_k1_MAT_s003
                                 + 0.937112 w_k1_MSFT_s003
                                 + 1.0022 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.795482 w_k1_ADM_s003
                                 + 1.17964 w_k1_AXP_s003
                                 + 0.758972 w_k1_DIS_s003
                                 + 0.884693 w_k1_FDX_s003
                                 + 1.32034 w_k1_IBM_s003
                                 + 1.08553 w_k1_INTC_s003
                                 + 0.97285 w_k1_KO_s003 + 1.90104 w_k1_MAT_s003
                                 + 0.94223 w_k1_MSFT_s003
                                 + 0.998998 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.72682 w_k1_ADM_s003
                                 + 0.629917 w_k1_AXP_s003
                                 + 0.630598 w_k1_DIS_s003
                                 + 0.785273 w_k1_FDX_s003
                                 + 0.707808 w_k1_IBM_s003
                                 + 0.915079 w_k1_INTC_s003
                                 + 0.638602 w_k1_KO_s003
                                 + 0.928642 w_k1_MAT_s003
                                 + 0.835154 w_k1_MSFT_s003
                                 + 0.901231 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.33616 w_k1_ADM_s003
                                 + 1.61025 w_k1_AXP_s003
                                 + 1.51183 w_k1_DIS_s003
                                 + 1.39977 w_k1_FDX_s003
                                 + 1.11415 w_k1_IBM_s003
                                 + 1.30412 w_k1_INTC_s003 + 1.1714 w_k1_KO_s003
                                 + 1.53895 w_k1_MAT_s003
                                 + 1.11497 w_k1_MSFT_s003
                                 + 1.26415 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.889364 w_k1_ADM_s003
                                 + 1.11872 w_k1_AXP_s003
                                 + 1.65498 w_k1_DIS_s003
                                 + 1.19186 w_k1_FDX_s003
                                 + 1.18152 w_k1_IBM_s003
                                 + 1.36206 w_k1_INTC_s003
                                 + 1.25299 w_k1_KO_s003 + 1.5158 w_k1_MAT_s003
                                 + 1.31871 w_k1_MSFT_s003
                                 + 1.20769 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.984976 w_k1_ADM_s003
                                 + 1.20671 w_k1_AXP_s003
                                 + 1.12611 w_k1_DIS_s003
                                 + 1.02002 w_k1_FDX_s003
                                 + 1.03844 w_k1_IBM_s003
                                 + 1.24531 w_k1_INTC_s003
                                 + 1.06996 w_k1_KO_s003 + 1.2648 w_k1_MAT_s003
                                 + 1.04575 w_k1_MSFT_s003
                                 + 1.00042 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.08351 w_k1_ADM_s004
                                 + 1.04407 w_k1_AXP_s004
                                 + 1.21285 w_k1_DIS_s004
                                 + 1.00689 w_k1_FDX_s004
                                 + 1.27385 w_k1_IBM_s004
                                 + 0.96303 w_k1_INTC_s004
                                 + 1.20508 w_k1_KO_s004 + 1.05861 w_k1_MAT_s004
                                 + 1.0784 w_k1_MSFT_s004 + 1.06431 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.999179 w_k1_ADM_s004
                                 + 0.963475 w_k1_AXP_s004
                                 + 0.951914 w_k1_DIS_s004
                                 + 0.940598 w_k1_FDX_s004
                                 + 1.14579 w_k1_IBM_s004
                                 + 0.930533 w_k1_INTC_s004
                                 + 1.13313 w_k1_KO_s004
                                 + 0.965762 w_k1_MAT_s004
                                 + 0.994113 w_k1_MSFT_s004
                                 + 1.05087 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.27642 w_k1_ADM_s004
                                 + 1.13736 w_k1_AXP_s004
                                 + 1.51176 w_k1_DIS_s004
                                 + 1.21719 w_k1_FDX_s004
                                 + 1.67092 w_k1_IBM_s004
                                 + 1.07855 w_k1_INTC_s004
                                 + 1.39843 w_k1_KO_s004 + 2.19341 w_k1_MAT_s004
                                 + 1.23218 w_k1_MSFT_s004
                                 + 1.24803 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.24395 w_k1_ADM_s004
                                 + 1.43676 w_k1_AXP_s004
                                 + 1.35647 w_k1_DIS_s004
                                 + 1.27943 w_k1_FDX_s004
                                 + 0.970346 w_k1_IBM_s004
                                 + 1.18173 w_k1_INTC_s004
                                 + 1.02665 w_k1_KO_s004
                                 + 0.968224 w_k1_MAT_s004
                                 + 1.34446 w_k1_MSFT_s004
                                 + 1.17071 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.826167 w_k1_ADM_s004
                                 + 0.787582 w_k1_AXP_s004
                                 + 0.661654 w_k1_DIS_s004
                                 + 0.780851 w_k1_FDX_s004
                                 + 0.851964 w_k1_IBM_s004
                                 + 0.844358 w_k1_INTC_s004
                                 + 0.741061 w_k1_KO_s004
                                 + 0.905367 w_k1_MAT_s004
                                 + 0.903819 w_k1_MSFT_s004
                                 + 0.892672 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.12302 w_k1_ADM_s004
                                 + 1.10821 w_k1_AXP_s004
                                 + 0.94502 w_k1_DIS_s004
                                 + 1.18873 w_k1_FDX_s004
                                 + 0.743955 w_k1_IBM_s004
                                 + 0.899197 w_k1_INTC_s004
                                 + 0.914514 w_k1_KO_s004
                                 + 1.29127 w_k1_MAT_s004
                                 + 1.02825 w_k1_MSFT_s004
                                 + 0.987157 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.49654 w_k1_ADM_s004
                                 + 1.4402 w_k1_AXP_s004 + 1.55158 w_k1_DIS_s004
                                 + 1.47117 w_k1_FDX_s004
                                 + 1.05784 w_k1_IBM_s004
                                 + 1.37047 w_k1_INTC_s004
                                 + 1.26072 w_k1_KO_s004 + 1.1138 w_k1_MAT_s004
                                 + 1.36547 w_k1_MSFT_s004
                                 + 1.16045 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.40792 w_k1_ADM_s004
                                 + 1.42405 w_k1_AXP_s004 + 1.0904 w_k1_DIS_s004
                                 + 1.07429 w_k1_FDX_s004
                                 + 1.34346 w_k1_IBM_s004
                                 + 0.816408 w_k1_INTC_s004
                                 + 1.09746 w_k1_KO_s004 + 1.59313 w_k1_MAT_s004
                                 + 0.991534 w_k1_MSFT_s004
                                 + 1.25738 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.754279 w_k1_ADM_s005
                                 + 0.887101 w_k1_AXP_s005
                                 + 1.01019 w_k1_DIS_s005
                                 + 0.845443 w_k1_FDX_s005
                                 + 1.51395 w_k1_IBM_s005
                                 + 0.862004 w_k1_INTC_s005
                                 + 1.16717 w_k1_KO_s005 + 1.50795 w_k1_MAT_s005
                                 + 0.943707 w_k1_MSFT_s005
                                 + 0.978967 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.742624 w_k1_ADM_s005
                                 + 0.652296 w_k1_AXP_s005
                                 + 0.671347 w_k1_DIS_s005
                                 + 0.774252 w_k1_FDX_s005
                                 + 1.04986 w_k1_IBM_s005
                                 + 0.855309 w_k1_INTC_s005
                                 + 0.887702 w_k1_KO_s005
                                 + 1.17937 w_k1_MAT_s005
                                 + 0.895104 w_k1_MSFT_s005
                                 + 0.847511 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.883345 w_k1_ADM_s005
                                 + 0.763243 w_k1_AXP_s005
                                 + 0.880062 w_k1_DIS_s005
                                 + 0.966627 w_k1_FDX_s005
                                 + 0.775961 w_k1_IBM_s005
                                 + 0.875428 w_k1_INTC_s005
                                 + 0.778662 w_k1_KO_s005
                                 + 0.763701 w_k1_MAT_s005
                                 + 0.967385 w_k1_MSFT_s005
                                 + 0.881407 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.29066 w_k1_ADM_s005
                                 + 1.00006 w_k1_AXP_s005
                                 + 1.25857 w_k1_DIS_s005
                                 + 1.19841 w_k1_FDX_s005
                                 + 1.23123 w_k1_IBM_s005
                                 + 0.827425 w_k1_INTC_s005
                                 + 1.16723 w_k1_KO_s005 + 1.16102 w_k1_MAT_s005
                                 + 1.26379 w_k1_MSFT_s005
                                 + 1.09217 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.999133 w_k1_ADM_s005
                                 + 0.721484 w_k1_AXP_s005
                                 + 1.04299 w_k1_DIS_s005
                                 + 1.06009 w_k1_FDX_s005
                                 + 0.667166 w_k1_IBM_s005
                                 + 0.906962 w_k1_INTC_s005
                                 + 0.838605 w_k1_KO_s005
                                 + 0.64617 w_k1_MAT_s005
                                 + 0.979929 w_k1_MSFT_s005
                                 + 0.981237 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.25784 w_k1_ADM_s005
                                 + 1.04696 w_k1_AXP_s005
                                 + 1.23279 w_k1_DIS_s005 + 1.1746 w_k1_FDX_s005
                                 + 0.914577 w_k1_IBM_s005
                                 + 1.17483 w_k1_INTC_s005
                                 + 1.07195 w_k1_KO_s005 + 1.01659 w_k1_MAT_s005
                                 + 1.34152 w_k1_MSFT_s005
                                 + 0.978259 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.37335 w_k1_ADM_s005
                                 + 1.24358 w_k1_AXP_s005
                                 + 1.63901 w_k1_DIS_s005
                                 + 1.29598 w_k1_FDX_s005
                                 + 1.62047 w_k1_IBM_s005
                                 + 1.18815 w_k1_INTC_s005 + 1.577 w_k1_KO_s005
                                 + 1.61323 w_k1_MAT_s005
                                 + 1.34026 w_k1_MSFT_s005
                                 + 1.19581 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.5435 w_k1_ADM_s005
                                 + 1.16303 w_k1_AXP_s005
                                 + 1.69289 w_k1_DIS_s005
                                 + 1.59766 w_k1_FDX_s005
                                 + 0.877906 w_k1_IBM_s005
                                 + 1.46446 w_k1_INTC_s005
                                 + 1.38599 w_k1_KO_s005
                                 + 0.665766 w_k1_MAT_s005
                                 + 1.55128 w_k1_MSFT_s005
                                 + 1.28255 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.879444 w_k1_ADM_s006
                                 + 1.21782 w_k1_AXP_s006
                                 + 0.905026 w_k1_DIS_s006
                                 + 0.910648 w_k1_FDX_s006
                                 + 1.5718 w_k1_IBM_s006
                                 + 1.14522 w_k1_INTC_s006
                                 + 0.976519 w_k1_KO_s006
                                 + 1.67271 w_k1_MAT_s006
                                 + 1.00688 w_k1_MSFT_s006 + 1.0347 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.05956 w_k1_ADM_s006
                                 + 1.14159 w_k1_AXP_s006
                                 + 0.961442 w_k1_DIS_s006
                                 + 0.982727 w_k1_FDX_s006
                                 + 0.77583 w_k1_IBM_s006
                                 + 0.826487 w_k1_INTC_s006
                                 + 0.75494 w_k1_KO_s006
                                 + 0.987079 w_k1_MAT_s006
                                 + 0.94454 w_k1_MSFT_s006
                                 + 0.935511 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.29329 w_k1_ADM_s006
                                 + 1.2477 w_k1_AXP_s006 + 1.24105 w_k1_DIS_s006
                                 + 1.22752 w_k1_FDX_s006
                                 + 0.744332 w_k1_IBM_s006
                                 + 1.17996 w_k1_INTC_s006
                                 + 0.974555 w_k1_KO_s006
                                 + 0.982372 w_k1_MAT_s006
                                 + 1.13732 w_k1_MSFT_s006
                                 + 1.13001 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.756527 w_k1_ADM_s006
                                 + 0.76213 w_k1_AXP_s006
                                 + 0.604183 w_k1_DIS_s006
                                 + 0.718716 w_k1_FDX_s006
                                 + 0.943789 w_k1_IBM_s006
                                 + 0.998691 w_k1_INTC_s006
                                 + 0.642766 w_k1_KO_s006
                                 + 0.844762 w_k1_MAT_s006
                                 + 0.821366 w_k1_MSFT_s006
                                 + 0.876544 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.03569 w_k1_ADM_s006
                                 + 1.13852 w_k1_AXP_s006
                                 + 0.989609 w_k1_DIS_s006
                                 + 0.988806 w_k1_FDX_s006
                                 + 1.03264 w_k1_IBM_s006
                                 + 0.901261 w_k1_INTC_s006
                                 + 0.765092 w_k1_KO_s006
                                 + 1.32938 w_k1_MAT_s006
                                 + 0.908303 w_k1_MSFT_s006
                                 + 1.00847 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.24044 w_k1_ADM_s006
                                 + 1.39542 w_k1_AXP_s006 + 1.4418 w_k1_DIS_s006
                                 + 1.26755 w_k1_FDX_s006
                                 + 1.33381 w_k1_IBM_s006
                                 + 1.26121 w_k1_INTC_s006
                                 + 1.20699 w_k1_KO_s006 + 1.3865 w_k1_MAT_s006
                                 + 1.21433 w_k1_MSFT_s006
                                 + 1.32546 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.966288 w_k1_ADM_s006
                                 + 0.807002 w_k1_AXP_s006
                                 + 0.801955 w_k1_DIS_s006
                                 + 0.932584 w_k1_FDX_s006
                                 + 0.653025 w_k1_IBM_s006
                                 + 1.13663 w_k1_INTC_s006 + 0.8161 w_k1_KO_s006
                                 + 0.524556 w_k1_MAT_s006
                                 + 1.01528 w_k1_MSFT_s006
                                 + 0.98476 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.828534 w_k1_ADM_s006
                                 + 0.848588 w_k1_AXP_s006
                                 + 0.762142 w_k1_DIS_s006
                                 + 0.825877 w_k1_FDX_s006
                                 + 0.815186 w_k1_IBM_s006
                                 + 0.835203 w_k1_INTC_s006
                                 + 0.72518 w_k1_KO_s006
                                 + 0.818996 w_k1_MAT_s006
                                 + 0.858124 w_k1_MSFT_s006
                                 + 0.893849 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.15797 w_k1_ADM_s007
                                 + 1.17071 w_k1_AXP_s007
                                 + 0.989543 w_k1_DIS_s007
                                 + 1.13396 w_k1_FDX_s007
                                 + 1.21635 w_k1_IBM_s007
                                 + 1.13546 w_k1_INTC_s007
                                 + 1.17808 w_k1_KO_s007 + 1.24868 w_k1_MAT_s007
                                 + 1.16601 w_k1_MSFT_s007
                                 + 1.04895 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.12521 w_k1_ADM_s007
                                 + 1.302 w_k1_AXP_s007 + 1.95193 w_k1_DIS_s007
                                 + 1.46797 w_k1_FDX_s007
                                 + 1.17693 w_k1_IBM_s007
                                 + 1.17852 w_k1_INTC_s007
                                 + 1.30234 w_k1_KO_s007 + 1.7318 w_k1_MAT_s007
                                 + 1.28105 w_k1_MSFT_s007
                                 + 1.22744 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.965769 w_k1_ADM_s007
                                 + 0.841547 w_k1_AXP_s007
                                 + 0.888626 w_k1_DIS_s007
                                 + 0.925458 w_k1_FDX_s007
                                 + 0.814955 w_k1_IBM_s007
                                 + 0.849078 w_k1_INTC_s007
                                 + 0.85452 w_k1_KO_s007
                                 + 0.709162 w_k1_MAT_s007
                                 + 0.829245 w_k1_MSFT_s007
                                 + 0.905963 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.23559 w_k1_ADM_s007
                                 + 1.31728 w_k1_AXP_s007
                                 + 1.27397 w_k1_DIS_s007
                                 + 1.36469 w_k1_FDX_s007
                                 + 1.10371 w_k1_IBM_s007
                                 + 1.30268 w_k1_INTC_s007
                                 + 1.14403 w_k1_KO_s007 + 1.13245 w_k1_MAT_s007
                                 + 1.17324 w_k1_MSFT_s007
                                 + 1.07175 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.69064 w_k1_ADM_s007
                                 + 0.850089 w_k1_AXP_s007
                                 + 0.916471 w_k1_DIS_s007
                                 + 0.838528 w_k1_FDX_s007
                                 + 1.34395 w_k1_IBM_s007
                                 + 0.947864 w_k1_INTC_s007
                                 + 1.09037 w_k1_KO_s007 + 1.07102 w_k1_MAT_s007
                                 + 0.913714 w_k1_MSFT_s007
                                 + 0.963791 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.786401 w_k1_ADM_s007
                                 + 0.774173 w_k1_AXP_s007
                                 + 0.69856 w_k1_DIS_s007
                                 + 0.780593 w_k1_FDX_s007
                                 + 0.740605 w_k1_IBM_s007
                                 + 0.774827 w_k1_INTC_s007
                                 + 0.736028 w_k1_KO_s007
                                 + 0.631052 w_k1_MAT_s007
                                 + 0.735497 w_k1_MSFT_s007
                                 + 0.85493 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.13528 w_k1_ADM_s007
                                 + 0.915873 w_k1_AXP_s007
                                 + 0.936765 w_k1_DIS_s007
                                 + 1.0216 w_k1_FDX_s007
                                 + 0.940726 w_k1_IBM_s007
                                 + 1.0632 w_k1_INTC_s007 + 1.05765 w_k1_KO_s007
                                 + 0.930654 w_k1_MAT_s007
                                 + 0.987911 w_k1_MSFT_s007
                                 + 1.0298 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.15279 w_k1_ADM_s007
                                 + 1.19363 w_k1_AXP_s007
                                 + 1.16731 w_k1_DIS_s007
                                 + 1.26498 w_k1_FDX_s007
                                 + 0.719966 w_k1_IBM_s007
                                 + 1.22516 w_k1_INTC_s007
                                 + 0.968416 w_k1_KO_s007
                                 + 0.623786 w_k1_MAT_s007
                                 + 1.11339 w_k1_MSFT_s007
                                 + 1.03246 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.0164 w_k1_ADM_s010
                                 + 1.18995 w_k1_AXP_s010
                                 + 0.922348 w_k1_DIS_s010
                                 + 1.04376 w_k1_FDX_s010
                                 + 0.925426 w_k1_IBM_s010
                                 + 1.29054 w_k1_INTC_s010
                                 + 1.12742 w_k1_KO_s010 + 1.20949 w_k1_MAT_s010
                                 + 1.17577 w_k1_MSFT_s010
                                 + 1.08005 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.29198 w_k1_ADM_s010
                                 + 1.15933 w_k1_AXP_s010
                                 + 1.44861 w_k1_DIS_s010
                                 + 1.23683 w_k1_FDX_s010
                                 + 1.24243 w_k1_IBM_s010
                                 + 1.11734 w_k1_INTC_s010
                                 + 1.28715 w_k1_KO_s010 + 1.3236 w_k1_MAT_s010
                                 + 1.37754 w_k1_MSFT_s010
                                 + 1.10988 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.47637 w_k1_ADM_s010
                                 + 1.70042 w_k1_AXP_s010
                                 + 1.32306 w_k1_DIS_s010
                                 + 1.46127 w_k1_FDX_s010 + 1.0568 w_k1_IBM_s010
                                 + 1.49033 w_k1_INTC_s010
                                 + 1.13816 w_k1_KO_s010 + 1.13967 w_k1_MAT_s010
                                 + 1.32537 w_k1_MSFT_s010
                                 + 1.08873 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.887645 w_k1_ADM_s010
                                 + 0.792674 w_k1_AXP_s010
                                 + 0.688897 w_k1_DIS_s010
                                 + 0.836997 w_k1_FDX_s010
                                 + 0.651387 w_k1_IBM_s010
                                 + 0.863735 w_k1_INTC_s010
                                 + 0.796648 w_k1_KO_s010
                                 + 0.599515 w_k1_MAT_s010
                                 + 0.830197 w_k1_MSFT_s010
                                 + 0.896663 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.787015 w_k1_ADM_s010
                                 + 0.893749 w_k1_AXP_s010
                                 + 0.673355 w_k1_DIS_s010
                                 + 0.853296 w_k1_FDX_s010
                                 + 1.00694 w_k1_IBM_s010
                                 + 0.978538 w_k1_INTC_s010
                                 + 0.862842 w_k1_KO_s010
                                 + 1.16783 w_k1_MAT_s010
                                 + 0.85171 w_k1_MSFT_s010
                                 + 0.903169 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.814527 w_k1_ADM_s010
                                 + 0.908741 w_k1_AXP_s010
                                 + 1.03042 w_k1_DIS_s010
                                 + 0.885961 w_k1_FDX_s010
                                 + 1.02728 w_k1_IBM_s010
                                 + 1.19875 w_k1_INTC_s010
                                 + 1.05691 w_k1_KO_s010
                                 + 0.902453 w_k1_MAT_s010
                                 + 1.04504 w_k1_MSFT_s010
                                 + 0.895284 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.26842 w_k1_ADM_s010
                                 + 0.97845 w_k1_AXP_s010
                                 + 1.07092 w_k1_DIS_s010
                                 + 1.18609 w_k1_FDX_s010
                                 + 0.665515 w_k1_IBM_s010
                                 + 1.23351 w_k1_INTC_s010
                                 + 0.927919 w_k1_KO_s010
                                 + 0.653361 w_k1_MAT_s010
                                 + 1.25835 w_k1_MSFT_s010
                                 + 1.04808 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.13257 w_k1_ADM_s010
                                 + 1.0283 w_k1_AXP_s010 + 1.2748 w_k1_DIS_s010
                                 + 1.08498 w_k1_FDX_s010
                                 + 1.57798 w_k1_IBM_s010
                                 + 0.937487 w_k1_INTC_s010
                                 + 1.31247 w_k1_KO_s010 + 1.24318 w_k1_MAT_s010
                                 + 1.07469 w_k1_MSFT_s010
                                 + 1.09678 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.944338 w_k1_ADM_s011
                                 + 0.697929 w_k1_AXP_s011
                                 + 0.544934 w_k1_DIS_s011
                                 + 0.660595 w_k1_FDX_s011
                                 + 0.806303 w_k1_IBM_s011
                                 + 0.998555 w_k1_INTC_s011
                                 + 0.719279 w_k1_KO_s011
                                 + 0.460034 w_k1_MAT_s011
                                 + 0.796956 w_k1_MSFT_s011
                                 + 0.846114 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.22721 w_k1_ADM_s011
                                 + 1.04965 w_k1_AXP_s011 + 1.8225 w_k1_DIS_s011
                                 + 1.36949 w_k1_FDX_s011
                                 + 1.15355 w_k1_IBM_s011
                                 + 1.16775 w_k1_INTC_s011
                                 + 1.51014 w_k1_KO_s011
                                 + 0.948413 w_k1_MAT_s011
                                 + 1.51783 w_k1_MSFT_s011
                                 + 1.35059 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.97638 w_k1_ADM_s011
                                 + 0.913421 w_k1_AXP_s011
                                 + 0.746833 w_k1_DIS_s011
                                 + 0.835882 w_k1_FDX_s011
                                 + 1.26584 w_k1_IBM_s011
                                 + 0.964787 w_k1_INTC_s011
                                 + 0.926639 w_k1_KO_s011
                                 + 1.10008 w_k1_MAT_s011
                                 + 0.995506 w_k1_MSFT_s011
                                 + 1.13668 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.52035 w_k1_ADM_s011
                                 + 1.4181 w_k1_AXP_s011 + 1.42838 w_k1_DIS_s011
                                 + 1.25346 w_k1_FDX_s011
                                 + 1.15862 w_k1_IBM_s011
                                 + 1.49227 w_k1_INTC_s011
                                 + 1.24192 w_k1_KO_s011
                                 + 0.964674 w_k1_MAT_s011
                                 + 1.31512 w_k1_MSFT_s011
                                 + 1.21018 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.884987 w_k1_ADM_s011
                                 + 0.842545 w_k1_AXP_s011
                                 + 0.801839 w_k1_DIS_s011
                                 + 0.909326 w_k1_FDX_s011
                                 + 1.01854 w_k1_IBM_s011
                                 + 1.02399 w_k1_INTC_s011
                                 + 0.926971 w_k1_KO_s011
                                 + 0.757918 w_k1_MAT_s011
                                 + 0.951368 w_k1_MSFT_s011
                                 + 0.940033 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.51149 w_k1_ADM_s011
                                 + 1.26902 w_k1_AXP_s011
                                 + 1.21139 w_k1_DIS_s011
                                 + 1.33615 w_k1_FDX_s011
                                 + 0.735677 w_k1_IBM_s011
                                 + 1.33734 w_k1_INTC_s011
                                 + 1.01558 w_k1_KO_s011
                                 + 0.572727 w_k1_MAT_s011
                                 + 1.2525 w_k1_MSFT_s011 + 1.1795 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.20777 w_k1_ADM_s011
                                 + 0.780946 w_k1_AXP_s011
                                 + 0.856728 w_k1_DIS_s011
                                 + 1.00494 w_k1_FDX_s011
                                 + 0.714516 w_k1_IBM_s011
                                 + 0.883345 w_k1_INTC_s011
                                 + 0.808228 w_k1_KO_s011
                                 + 0.558449 w_k1_MAT_s011
                                 + 0.954276 w_k1_MSFT_s011
                                 + 0.953691 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09927 w_k1_ADM_s011
                                 + 1.07603 w_k1_AXP_s011 + 1.0462 w_k1_DIS_s011
                                 + 0.966434 w_k1_FDX_s011
                                 + 1.49964 w_k1_IBM_s011
                                 + 1.0628 w_k1_INTC_s011 + 1.12436 w_k1_KO_s011
                                 + 1.23363 w_k1_MAT_s011
                                 + 0.99255 w_k1_MSFT_s011
                                 + 1.23153 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.955855 w_k1_ADM_s012
                                 + 1.3115 w_k1_AXP_s012 + 1.46308 w_k1_DIS_s012
                                 + 1.18607 w_k1_FDX_s012
                                 + 1.66824 w_k1_IBM_s012
                                 + 1.13511 w_k1_INTC_s012
                                 + 1.52068 w_k1_KO_s012 + 1.45485 w_k1_MAT_s012
                                 + 1.25769 w_k1_MSFT_s012
                                 + 1.35978 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.28166 w_k1_ADM_s012
                                 + 1.75544 w_k1_AXP_s012
                                 + 1.42042 w_k1_DIS_s012
                                 + 1.35914 w_k1_FDX_s012
                                 + 1.00862 w_k1_IBM_s012
                                 + 1.22341 w_k1_INTC_s012
                                 + 1.17548 w_k1_KO_s012 + 1.25528 w_k1_MAT_s012
                                 + 1.1982 w_k1_MSFT_s012 + 1.29073 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.697755 w_k1_ADM_s012
                                 + 0.708092 w_k1_AXP_s012
                                 + 0.697925 w_k1_DIS_s012
                                 + 0.7668 w_k1_FDX_s012
                                 + 0.803837 w_k1_IBM_s012
                                 + 0.901337 w_k1_INTC_s012
                                 + 0.892094 w_k1_KO_s012
                                 + 0.782077 w_k1_MAT_s012
                                 + 0.841619 w_k1_MSFT_s012
                                 + 0.88452 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.837564 w_k1_ADM_s012
                                 + 0.982693 w_k1_AXP_s012
                                 + 0.832055 w_k1_DIS_s012
                                 + 0.913994 w_k1_FDX_s012
                                 + 1.05601 w_k1_IBM_s012
                                 + 0.932374 w_k1_INTC_s012
                                 + 1.00113 w_k1_KO_s012 + 1.01372 w_k1_MAT_s012
                                 + 0.842624 w_k1_MSFT_s012
                                 + 0.983564 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.872913 w_k1_ADM_s012
                                 + 1.23556 w_k1_AXP_s012
                                 + 1.09444 w_k1_DIS_s012
                                 + 0.948834 w_k1_FDX_s012
                                 + 1.37869 w_k1_IBM_s012
                                 + 0.996933 w_k1_INTC_s012
                                 + 1.21941 w_k1_KO_s012 + 1.66443 w_k1_MAT_s012
                                 + 1.09444 w_k1_MSFT_s012
                                 + 1.12565 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.819174 w_k1_ADM_s012
                                 + 0.881856 w_k1_AXP_s012
                                 + 0.582047 w_k1_DIS_s012
                                 + 0.82409 w_k1_FDX_s012
                                 + 0.705122 w_k1_IBM_s012
                                 + 0.807074 w_k1_INTC_s012
                                 + 0.777296 w_k1_KO_s012
                                 + 0.821684 w_k1_MAT_s012
                                 + 0.865926 w_k1_MSFT_s012
                                 + 0.788678 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.796692 w_k1_ADM_s012
                                 + 1.01274 w_k1_AXP_s012
                                 + 0.841468 w_k1_DIS_s012
                                 + 1.01223 w_k1_FDX_s012
                                 + 0.70148 w_k1_IBM_s012
                                 + 1.08022 w_k1_INTC_s012
                                 + 0.804725 w_k1_KO_s012
                                 + 0.719244 w_k1_MAT_s012
                                 + 0.90122 w_k1_MSFT_s012
                                 + 0.907096 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.05148 w_k1_ADM_s012
                                 + 1.09848 w_k1_AXP_s012
                                 + 1.24869 w_k1_DIS_s012
                                 + 1.13503 w_k1_FDX_s012
                                 + 0.987154 w_k1_IBM_s012
                                 + 1.1016 w_k1_INTC_s012 + 1.10931 w_k1_KO_s012
                                 + 1.05943 w_k1_MAT_s012
                                 + 1.20596 w_k1_MSFT_s012 + 1.1176 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.585504 w_k1_ADM_s013
                                 + 0.623726 w_k1_AXP_s013
                                 + 0.803848 w_k1_DIS_s013
                                 + 0.854999 w_k1_FDX_s013
                                 + 0.636616 w_k1_IBM_s013
                                 + 0.6995 w_k1_INTC_s013
                                 + 0.620609 w_k1_KO_s013
                                 + 0.681019 w_k1_MAT_s013
                                 + 0.827153 w_k1_MSFT_s013
                                 + 0.828874 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.773632 w_k1_ADM_s013
                                 + 0.542401 w_k1_AXP_s013
                                 + 0.693071 w_k1_DIS_s013
                                 + 0.576857 w_k1_FDX_s013
                                 + 1.19696 w_k1_IBM_s013
                                 + 0.606253 w_k1_INTC_s013
                                 + 1.00105 w_k1_KO_s013 + 1.37126 w_k1_MAT_s013
                                 + 0.613305 w_k1_MSFT_s013
                                 + 0.854387 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.02553 w_k1_ADM_s013
                                 + 1.36566 w_k1_AXP_s013
                                 + 1.35858 w_k1_DIS_s013
                                 + 1.06935 w_k1_FDX_s013
                                 + 1.83832 w_k1_IBM_s013
                                 + 0.904505 w_k1_INTC_s013
                                 + 1.22285 w_k1_KO_s013 + 1.82339 w_k1_MAT_s013
                                 + 0.943462 w_k1_MSFT_s013
                                 + 1.07333 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.772912 w_k1_ADM_s013
                                 + 0.626412 w_k1_AXP_s013
                                 + 0.800183 w_k1_DIS_s013
                                 + 0.762708 w_k1_FDX_s013
                                 + 0.959491 w_k1_IBM_s013
                                 + 0.846951 w_k1_INTC_s013
                                 + 0.830014 w_k1_KO_s013
                                 + 0.581727 w_k1_MAT_s013
                                 + 0.780484 w_k1_MSFT_s013
                                 + 0.833126 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.14956 w_k1_ADM_s013
                                 + 0.984915 w_k1_AXP_s013
                                 + 0.947717 w_k1_DIS_s013
                                 + 1.09911 w_k1_FDX_s013
                                 + 0.792343 w_k1_IBM_s013
                                 + 0.829482 w_k1_INTC_s013
                                 + 0.891568 w_k1_KO_s013
                                 + 1.23832 w_k1_MAT_s013
                                 + 0.923626 w_k1_MSFT_s013
                                 + 1.02644 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.04907 w_k1_ADM_s013
                                 + 0.911669 w_k1_AXP_s013
                                 + 1.01286 w_k1_DIS_s013
                                 + 0.908463 w_k1_FDX_s013
                                 + 1.07747 w_k1_IBM_s013
                                 + 0.829681 w_k1_INTC_s013
                                 + 0.946974 w_k1_KO_s013
                                 + 1.54554 w_k1_MAT_s013
                                 + 0.92799 w_k1_MSFT_s013
                                 + 0.941587 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.03953 w_k1_ADM_s013
                                 + 1.06239 w_k1_AXP_s013
                                 + 1.14322 w_k1_DIS_s013
                                 + 1.01284 w_k1_FDX_s013
                                 + 1.34801 w_k1_IBM_s013
                                 + 1.08236 w_k1_INTC_s013
                                 + 1.26192 w_k1_KO_s013 + 1.52593 w_k1_MAT_s013
                                 + 0.998603 w_k1_MSFT_s013
                                 + 1.20431 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.08547 w_k1_ADM_s013
                                 + 1.21852 w_k1_AXP_s013
                                 + 1.67038 w_k1_DIS_s013
                                 + 1.32198 w_k1_FDX_s013
                                 + 0.943239 w_k1_IBM_s013
                                 + 1.08198 w_k1_INTC_s013
                                 + 1.29161 w_k1_KO_s013
                                 + 0.778807 w_k1_MAT_s013
                                 + 1.24199 w_k1_MSFT_s013
                                 + 1.14203 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.802089 w_k1_ADM_s014
                                 + 0.858491 w_k1_AXP_s014
                                 + 0.991841 w_k1_DIS_s014
                                 + 0.88791 w_k1_FDX_s014
                                 + 0.969309 w_k1_IBM_s014
                                 + 0.88271 w_k1_INTC_s014
                                 + 0.930149 w_k1_KO_s014
                                 + 1.12836 w_k1_MAT_s014
                                 + 0.869914 w_k1_MSFT_s014
                                 + 1.00841 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.11076 w_k1_ADM_s014
                                 + 1.26964 w_k1_AXP_s014
                                 + 1.10614 w_k1_DIS_s014
                                 + 1.18779 w_k1_FDX_s014
                                 + 1.28841 w_k1_IBM_s014
                                 + 1.13915 w_k1_INTC_s014
                                 + 0.971062 w_k1_KO_s014
                                 + 1.66726 w_k1_MAT_s014
                                 + 0.942684 w_k1_MSFT_s014
                                 + 1.0716 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.860542 w_k1_ADM_s014
                                 + 0.940949 w_k1_AXP_s014
                                 + 1.1239 w_k1_DIS_s014 + 1.07944 w_k1_FDX_s014
                                 + 0.975557 w_k1_IBM_s014
                                 + 1.22061 w_k1_INTC_s014
                                 + 1.04321 w_k1_KO_s014 + 1.20623 w_k1_MAT_s014
                                 + 1.07183 w_k1_MSFT_s014
                                 + 0.957026 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.795728 w_k1_ADM_s014
                                 + 0.910081 w_k1_AXP_s014
                                 + 0.802255 w_k1_DIS_s014
                                 + 0.822731 w_k1_FDX_s014
                                 + 1.25389 w_k1_IBM_s014
                                 + 1.03578 w_k1_INTC_s014
                                 + 0.913997 w_k1_KO_s014
                                 + 1.29594 w_k1_MAT_s014
                                 + 0.848904 w_k1_MSFT_s014
                                 + 0.908496 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.04319 w_k1_ADM_s014
                                 + 0.915254 w_k1_AXP_s014
                                 + 1.20644 w_k1_DIS_s014
                                 + 1.20576 w_k1_FDX_s014
                                 + 0.785329 w_k1_IBM_s014
                                 + 1.80037 w_k1_INTC_s014
                                 + 1.04869 w_k1_KO_s014
                                 + 0.760414 w_k1_MAT_s014
                                 + 1.25122 w_k1_MSFT_s014
                                 + 1.09294 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.804653 w_k1_ADM_s014
                                 + 1.09383 w_k1_AXP_s014
                                 + 0.889233 w_k1_DIS_s014
                                 + 0.97111 w_k1_FDX_s014
                                 + 1.38874 w_k1_IBM_s014
                                 + 1.34022 w_k1_INTC_s014
                                 + 1.20385 w_k1_KO_s014 + 1.49823 w_k1_MAT_s014
                                 + 0.924634 w_k1_MSFT_s014
                                 + 1.03695 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.22247 w_k1_ADM_s014
                                 + 1.33711 w_k1_AXP_s014 + 1.6288 w_k1_DIS_s014
                                 + 1.45929 w_k1_FDX_s014
                                 + 1.17318 w_k1_IBM_s014 + 1.464 w_k1_INTC_s014
                                 + 1.22916 w_k1_KO_s014 + 1.01267 w_k1_MAT_s014
                                 + 1.29944 w_k1_MSFT_s014
                                 + 1.17524 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.629132 w_k1_ADM_s014
                                 + 0.85284 w_k1_AXP_s014
                                 + 0.644814 w_k1_DIS_s014
                                 + 0.878037 w_k1_FDX_s014
                                 + 0.741085 w_k1_IBM_s014
                                 + 1.16987 w_k1_INTC_s014
                                 + 0.812918 w_k1_KO_s014
                                 + 0.686473 w_k1_MAT_s014
                                 + 0.798144 w_k1_MSFT_s014
                                 + 0.830442 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.801131 w_k1_ADM_s015
                                 + 1.09028 w_k1_AXP_s015
                                 + 0.948771 w_k1_DIS_s015
                                 + 1.12564 w_k1_FDX_s015
                                 + 0.88569 w_k1_IBM_s015
                                 + 1.32051 w_k1_INTC_s015
                                 + 0.971716 w_k1_KO_s015
                                 + 1.02097 w_k1_MAT_s015
                                 + 1.08734 w_k1_MSFT_s015
                                 + 1.08501 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.989649 w_k1_ADM_s015
                                 + 1.31927 w_k1_AXP_s015
                                 + 1.59097 w_k1_DIS_s015
                                 + 1.27079 w_k1_FDX_s015
                                 + 1.22113 w_k1_IBM_s015
                                 + 1.24848 w_k1_INTC_s015
                                 + 1.29809 w_k1_KO_s015 + 1.37355 w_k1_MAT_s015
                                 + 1.10821 w_k1_MSFT_s015
                                 + 1.30673 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.438928 w_k1_ADM_s015
                                 + 0.668047 w_k1_AXP_s015
                                 + 0.598529 w_k1_DIS_s015
                                 + 0.716615 w_k1_FDX_s015
                                 + 0.72336 w_k1_IBM_s015
                                 + 0.89601 w_k1_INTC_s015
                                 + 0.740009 w_k1_KO_s015
                                 + 0.845459 w_k1_MAT_s015
                                 + 0.798684 w_k1_MSFT_s015
                                 + 0.836216 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.766223 w_k1_ADM_s015
                                 + 0.897818 w_k1_AXP_s015
                                 + 1.0843 w_k1_DIS_s015 + 0.96503 w_k1_FDX_s015
                                 + 0.98838 w_k1_IBM_s015
                                 + 1.02869 w_k1_INTC_s015
                                 + 1.04558 w_k1_KO_s015 + 1.04579 w_k1_MAT_s015
                                 + 0.848751 w_k1_MSFT_s015
                                 + 1.13723 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.904732 w_k1_ADM_s015
                                 + 0.79349 w_k1_AXP_s015
                                 + 1.05935 w_k1_DIS_s015
                                 + 1.06813 w_k1_FDX_s015
                                 + 0.741925 w_k1_IBM_s015
                                 + 1.3721 w_k1_INTC_s015 + 1.04211 w_k1_KO_s015
                                 + 0.711122 w_k1_MAT_s015
                                 + 1.18987 w_k1_MSFT_s015
                                 + 0.858964 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.696523 w_k1_ADM_s015
                                 + 0.690861 w_k1_AXP_s015
                                 + 0.736112 w_k1_DIS_s015
                                 + 0.76481 w_k1_FDX_s015
                                 + 1.15148 w_k1_IBM_s015
                                 + 0.799413 w_k1_INTC_s015
                                 + 0.953102 w_k1_KO_s015
                                 + 0.863538 w_k1_MAT_s015
                                 + 0.762111 w_k1_MSFT_s015
                                 + 1.05828 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.962155 w_k1_ADM_s015
                                 + 1.20548 w_k1_AXP_s015
                                 + 0.975777 w_k1_DIS_s015 + 1.174 w_k1_FDX_s015
                                 + 0.821655 w_k1_IBM_s015
                                 + 1.20411 w_k1_INTC_s015
                                 + 0.914266 w_k1_KO_s015
                                 + 1.21028 w_k1_MAT_s015
                                 + 0.932643 w_k1_MSFT_s015
                                 + 0.93789 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.793356 w_k1_ADM_s015
                                 + 1.01957 w_k1_AXP_s015
                                 + 1.11557 w_k1_DIS_s015
                                 + 0.983022 w_k1_FDX_s015
                                 + 1.51041 w_k1_IBM_s015
                                 + 1.10397 w_k1_INTC_s015
                                 + 1.29687 w_k1_KO_s015 + 1.40823 w_k1_MAT_s015
                                 + 0.873832 w_k1_MSFT_s015
                                 + 1.19762 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.13123 w_k1_ADM_s016
                                 + 1.12404 w_k1_AXP_s016
                                 + 1.52308 w_k1_DIS_s016
                                 + 1.37823 w_k1_FDX_s016
                                 + 0.72595 w_k1_IBM_s016
                                 + 1.07414 w_k1_INTC_s016
                                 + 1.09677 w_k1_KO_s016
                                 + 0.937278 w_k1_MAT_s016
                                 + 1.30019 w_k1_MSFT_s016
                                 + 1.14369 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.748015 w_k1_ADM_s016
                                 + 0.902623 w_k1_AXP_s016
                                 + 0.820906 w_k1_DIS_s016
                                 + 0.904877 w_k1_FDX_s016
                                 + 1.04663 w_k1_IBM_s016
                                 + 0.751015 w_k1_INTC_s016
                                 + 0.915355 w_k1_KO_s016
                                 + 0.885362 w_k1_MAT_s016
                                 + 0.772155 w_k1_MSFT_s016
                                 + 0.87924 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.830407 w_k1_ADM_s016
                                 + 1.92427 w_k1_AXP_s016
                                 + 1.70197 w_k1_DIS_s016 + 1.2585 w_k1_FDX_s016
                                 + 1.68446 w_k1_IBM_s016
                                 + 1.10278 w_k1_INTC_s016
                                 + 1.44537 w_k1_KO_s016 + 1.96671 w_k1_MAT_s016
                                 + 1.04295 w_k1_MSFT_s016
                                 + 1.13111 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.832782 w_k1_ADM_s016
                                 + 0.729002 w_k1_AXP_s016
                                 + 0.594013 w_k1_DIS_s016
                                 + 0.660331 w_k1_FDX_s016
                                 + 0.898543 w_k1_IBM_s016
                                 + 0.598147 w_k1_INTC_s016
                                 + 0.749016 w_k1_KO_s016
                                 + 0.82807 w_k1_MAT_s016
                                 + 0.638334 w_k1_MSFT_s016
                                 + 0.797557 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.550025 w_k1_ADM_s016
                                 + 0.504817 w_k1_AXP_s016
                                 + 0.679744 w_k1_DIS_s016
                                 + 0.67348 w_k1_FDX_s016 + 0.6132 w_k1_IBM_s016
                                 + 0.714205 w_k1_INTC_s016
                                 + 0.740138 w_k1_KO_s016
                                 + 0.60086 w_k1_MAT_s016
                                 + 0.732346 w_k1_MSFT_s016
                                 + 0.7456 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.529538 w_k1_ADM_s016
                                 + 0.909258 w_k1_AXP_s016
                                 + 0.770577 w_k1_DIS_s016
                                 + 0.719837 w_k1_FDX_s016
                                 + 1.39189 w_k1_IBM_s016
                                 + 0.730885 w_k1_INTC_s016
                                 + 0.926672 w_k1_KO_s016
                                 + 1.26772 w_k1_MAT_s016
                                 + 0.666759 w_k1_MSFT_s016
                                 + 0.820716 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.996641 w_k1_ADM_s016
                                 + 0.824916 w_k1_AXP_s016
                                 + 0.734812 w_k1_DIS_s016
                                 + 0.867955 w_k1_FDX_s016
                                 + 0.719447 w_k1_IBM_s016
                                 + 0.673217 w_k1_INTC_s016
                                 + 0.760885 w_k1_KO_s016
                                 + 0.733615 w_k1_MAT_s016
                                 + 0.791468 w_k1_MSFT_s016
                                 + 0.869504 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.10855 w_k1_ADM_s016
                                 + 1.34788 w_k1_AXP_s016
                                 + 1.59634 w_k1_DIS_s016
                                 + 1.20831 w_k1_FDX_s016
                                 + 1.04864 w_k1_IBM_s016
                                 + 0.959409 w_k1_INTC_s016
                                 + 1.27785 w_k1_KO_s016 + 1.03827 w_k1_MAT_s016
                                 + 1.09335 w_k1_MSFT_s016
                                 + 1.18852 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.67066 w_k1_ADM_s017
                                 + 1.34295 w_k1_AXP_s017
                                 + 1.72173 w_k1_DIS_s017
                                 + 1.40251 w_k1_FDX_s017
                                 + 1.20971 w_k1_IBM_s017
                                 + 1.45359 w_k1_INTC_s017
                                 + 1.66638 w_k1_KO_s017 + 1.34025 w_k1_MAT_s017
                                 + 1.44331 w_k1_MSFT_s017
                                 + 1.17168 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.926117 w_k1_ADM_s017
                                 + 0.987905 w_k1_AXP_s017
                                 + 0.980398 w_k1_DIS_s017
                                 + 0.80438 w_k1_FDX_s017
                                 + 1.49741 w_k1_IBM_s017
                                 + 1.05454 w_k1_INTC_s017
                                 + 1.36894 w_k1_KO_s017 + 1.25996 w_k1_MAT_s017
                                 + 1.06135 w_k1_MSFT_s017
                                 + 0.994926 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.08478 w_k1_ADM_s017
                                 + 1.31864 w_k1_AXP_s017
                                 + 1.29894 w_k1_DIS_s017
                                 + 1.18095 w_k1_FDX_s017
                                 + 1.47724 w_k1_IBM_s017
                                 + 1.45052 w_k1_INTC_s017
                                 + 1.30782 w_k1_KO_s017 + 1.33549 w_k1_MAT_s017
                                 + 1.28381 w_k1_MSFT_s017
                                 + 1.05682 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.03835 w_k1_ADM_s017
                                 + 0.799692 w_k1_AXP_s017
                                 + 0.945513 w_k1_DIS_s017
                                 + 1.01409 w_k1_FDX_s017
                                 + 0.663117 w_k1_IBM_s017
                                 + 1.01016 w_k1_INTC_s017
                                 + 0.94779 w_k1_KO_s017
                                 + 0.576107 w_k1_MAT_s017
                                 + 0.994394 w_k1_MSFT_s017
                                 + 0.963631 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.12942 w_k1_ADM_s017
                                 + 1.13987 w_k1_AXP_s017
                                 + 1.04329 w_k1_DIS_s017
                                 + 1.02777 w_k1_FDX_s017
                                 + 1.19768 w_k1_IBM_s017
                                 + 1.26171 w_k1_INTC_s017 + 1.1254 w_k1_KO_s017
                                 + 1.07695 w_k1_MAT_s017
                                 + 1.11631 w_k1_MSFT_s017
                                 + 0.950422 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.07803 w_k1_ADM_s017
                                 + 0.982333 w_k1_AXP_s017
                                 + 0.851382 w_k1_DIS_s017
                                 + 0.98017 w_k1_FDX_s017
                                 + 0.973263 w_k1_IBM_s017
                                 + 0.897507 w_k1_INTC_s017
                                 + 0.850706 w_k1_KO_s017
                                 + 0.745727 w_k1_MAT_s017
                                 + 0.885264 w_k1_MSFT_s017
                                 + 0.838269 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.702171 w_k1_ADM_s017
                                 + 0.658273 w_k1_AXP_s017
                                 + 0.654506 w_k1_DIS_s017
                                 + 0.775918 w_k1_FDX_s017
                                 + 0.824529 w_k1_IBM_s017
                                 + 0.941101 w_k1_INTC_s017
                                 + 0.742164 w_k1_KO_s017
                                 + 0.570364 w_k1_MAT_s017
                                 + 0.864808 w_k1_MSFT_s017
                                 + 0.818944 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.972513 w_k1_ADM_s017
                                 + 0.864502 w_k1_AXP_s017
                                 + 1.23855 w_k1_DIS_s017
                                 + 1.08284 w_k1_FDX_s017
                                 + 1.06563 w_k1_IBM_s017
                                 + 1.4637 w_k1_INTC_s017 + 1.21104 w_k1_KO_s017
                                 + 0.624619 w_k1_MAT_s017
                                 + 1.26691 w_k1_MSFT_s017
                                 + 1.03446 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.25342 w_k1_ADM_s100
                                 + 1.73859 w_k1_AXP_s100
                                 + 1.69764 w_k1_DIS_s100
                                 + 1.25929 w_k1_FDX_s100
                                 + 1.07664 w_k1_IBM_s100
                                 + 1.12099 w_k1_INTC_s100 + 1.2218 w_k1_KO_s100
                                 + 1.21361 w_k1_MAT_s100
                                 + 1.17006 w_k1_MSFT_s100
                                 + 1.10191 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.01419 w_k1_ADM_s100
                                 + 0.771833 w_k1_AXP_s100
                                 + 0.683212 w_k1_DIS_s100
                                 + 0.84774 w_k1_FDX_s100 + 0.6737 w_k1_IBM_s100
                                 + 0.750245 w_k1_INTC_s100
                                 + 0.785255 w_k1_KO_s100
                                 + 0.735093 w_k1_MAT_s100
                                 + 0.802241 w_k1_MSFT_s100
                                 + 0.840175 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.75671 w_k1_ADM_s100
                                 + 0.806891 w_k1_AXP_s100
                                 + 0.66975 w_k1_DIS_s100
                                 + 0.694743 w_k1_FDX_s100
                                 + 0.862313 w_k1_IBM_s100
                                 + 0.812235 w_k1_INTC_s100
                                 + 0.832216 w_k1_KO_s100
                                 + 0.880837 w_k1_MAT_s100
                                 + 0.70278 w_k1_MSFT_s100
                                 + 0.880331 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.35341 w_k1_ADM_s100
                                 + 1.22681 w_k1_AXP_s100
                                 + 1.35694 w_k1_DIS_s100 + 1.3133 w_k1_FDX_s100
                                 + 0.812193 w_k1_IBM_s100
                                 + 1.14182 w_k1_INTC_s100
                                 + 0.936207 w_k1_KO_s100
                                 + 0.79887 w_k1_MAT_s100
                                 + 1.20936 w_k1_MSFT_s100
                                 + 1.03643 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.01122 w_k1_ADM_s100
                                 + 1.69139 w_k1_AXP_s100
                                 + 1.48461 w_k1_DIS_s100
                                 + 1.13396 w_k1_FDX_s100 + 1.6648 w_k1_IBM_s100
                                 + 1.26431 w_k1_INTC_s100
                                 + 1.30225 w_k1_KO_s100 + 1.80856 w_k1_MAT_s100
                                 + 1.17816 w_k1_MSFT_s100
                                 + 1.01909 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.968351 w_k1_ADM_s100
                                 + 0.941483 w_k1_AXP_s100
                                 + 1.11809 w_k1_DIS_s100
                                 + 0.977141 w_k1_FDX_s100
                                 + 1.02241 w_k1_IBM_s100
                                 + 0.92279 w_k1_INTC_s100
                                 + 1.10314 w_k1_KO_s100 + 1.34074 w_k1_MAT_s100
                                 + 1.08844 w_k1_MSFT_s100
                                 + 1.03258 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.17667 w_k1_ADM_s100
                                 + 1.13152 w_k1_AXP_s100
                                 + 0.993146 w_k1_DIS_s100
                                 + 1.07869 w_k1_FDX_s100
                                 + 0.705001 w_k1_IBM_s100
                                 + 1.12459 w_k1_INTC_s100
                                 + 0.945722 w_k1_KO_s100
                                 + 1.06321 w_k1_MAT_s100
                                 + 1.00404 w_k1_MSFT_s100
                                 + 0.975436 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.903923 w_k1_ADM_s100
                                 + 0.859102 w_k1_AXP_s100
                                 + 1.08673 w_k1_DIS_s100
                                 + 0.879709 w_k1_FDX_s100
                                 + 1.31029 w_k1_IBM_s100
                                 + 0.953987 w_k1_INTC_s100
                                 + 0.989074 w_k1_KO_s100
                                 + 1.28681 w_k1_MAT_s100
                                 + 0.909095 w_k1_MSFT_s100
                                 + 0.973974 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.865602 w_k1_ADM_s101
                                 + 0.851166 w_k1_AXP_s101
                                 + 0.689655 w_k1_DIS_s101
                                 + 0.807474 w_k1_FDX_s101
                                 + 1.11324 w_k1_IBM_s101
                                 + 1.03934 w_k1_INTC_s101
                                 + 0.932119 w_k1_KO_s101
                                 + 1.47849 w_k1_MAT_s101
                                 + 0.884593 w_k1_MSFT_s101
                                 + 0.854962 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.947634 w_k1_ADM_s101
                                 + 1.14508 w_k1_AXP_s101
                                 + 1.05231 w_k1_DIS_s101
                                 + 1.06335 w_k1_FDX_s101
                                 + 0.822071 w_k1_IBM_s101
                                 + 1.13813 w_k1_INTC_s101
                                 + 0.88674 w_k1_KO_s101 + 1.09418 w_k1_MAT_s101
                                 + 1.10113 w_k1_MSFT_s101
                                 + 0.934193 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.941868 w_k1_ADM_s101
                                 + 1.46224 w_k1_AXP_s101 + 1.3214 w_k1_DIS_s101
                                 + 1.13173 w_k1_FDX_s101
                                 + 1.30708 w_k1_IBM_s101
                                 + 1.24832 w_k1_INTC_s101
                                 + 1.11544 w_k1_KO_s101 + 1.93398 w_k1_MAT_s101
                                 + 1.11011 w_k1_MSFT_s101
                                 + 1.08796 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09803 w_k1_ADM_s101
                                 + 1.28606 w_k1_AXP_s101
                                 + 1.16003 w_k1_DIS_s101
                                 + 0.924252 w_k1_FDX_s101
                                 + 1.5723 w_k1_IBM_s101
                                 + 1.14509 w_k1_INTC_s101
                                 + 1.40312 w_k1_KO_s101 + 1.55522 w_k1_MAT_s101
                                 + 1.0203 w_k1_MSFT_s101 + 1.07755 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.692746 w_k1_ADM_s101
                                 + 0.98749 w_k1_AXP_s101
                                 + 0.838473 w_k1_DIS_s101
                                 + 0.814216 w_k1_FDX_s101
                                 + 0.696143 w_k1_IBM_s101
                                 + 0.961544 w_k1_INTC_s101
                                 + 0.698599 w_k1_KO_s101
                                 + 1.01613 w_k1_MAT_s101
                                 + 0.859023 w_k1_MSFT_s101
                                 + 0.842393 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.31799 w_k1_ADM_s101
                                 + 1.59188 w_k1_AXP_s101
                                 + 1.72445 w_k1_DIS_s101
                                 + 1.42153 w_k1_FDX_s101
                                 + 1.22579 w_k1_IBM_s101
                                 + 1.41735 w_k1_INTC_s101
                                 + 1.30843 w_k1_KO_s101 + 1.77109 w_k1_MAT_s101
                                 + 1.52996 w_k1_MSFT_s101
                                 + 1.28451 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.683746 w_k1_ADM_s101
                                 + 0.50674 w_k1_AXP_s101
                                 + 0.587519 w_k1_DIS_s101
                                 + 0.669773 w_k1_FDX_s101
                                 + 0.693171 w_k1_IBM_s101
                                 + 0.698221 w_k1_INTC_s101
                                 + 0.637784 w_k1_KO_s101
                                 + 0.553833 w_k1_MAT_s101
                                 + 0.749734 w_k1_MSFT_s101
                                 + 0.734683 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.928957 w_k1_ADM_s101
                                 + 1.1289 w_k1_AXP_s101 + 1.10687 w_k1_DIS_s101
                                 + 0.903524 w_k1_FDX_s101
                                 + 1.15959 w_k1_IBM_s101
                                 + 1.04515 w_k1_INTC_s101
                                 + 1.12934 w_k1_KO_s101 + 1.52716 w_k1_MAT_s101
                                 + 1.11102 w_k1_MSFT_s101
                                 + 1.07774 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.897229 w_k1_ADM_s102
                                 + 0.838375 w_k1_AXP_s102
                                 + 1.05465 w_k1_DIS_s102
                                 + 0.886764 w_k1_FDX_s102
                                 + 0.96586 w_k1_IBM_s102
                                 + 0.825015 w_k1_INTC_s102
                                 + 0.93511 w_k1_KO_s102 + 1.25053 w_k1_MAT_s102
                                 + 0.981088 w_k1_MSFT_s102
                                 + 0.967842 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.962157 w_k1_ADM_s102
                                 + 0.808384 w_k1_AXP_s102
                                 + 0.767494 w_k1_DIS_s102
                                 + 0.925496 w_k1_FDX_s102
                                 + 0.875833 w_k1_IBM_s102
                                 + 0.842541 w_k1_INTC_s102
                                 + 0.914594 w_k1_KO_s102
                                 + 1.38757 w_k1_MAT_s102
                                 + 1.01606 w_k1_MSFT_s102
                                 + 0.909314 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.33999 w_k1_ADM_s102
                                 + 1.32383 w_k1_AXP_s102
                                 + 1.61151 w_k1_DIS_s102
                                 + 1.15145 w_k1_FDX_s102
                                 + 1.30435 w_k1_IBM_s102
                                 + 1.02137 w_k1_INTC_s102
                                 + 1.32957 w_k1_KO_s102 + 1.86596 w_k1_MAT_s102
                                 + 1.26198 w_k1_MSFT_s102
                                 + 1.16181 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.04932 w_k1_ADM_s102
                                 + 1.53982 w_k1_AXP_s102
                                 + 1.04258 w_k1_DIS_s102
                                 + 0.955841 w_k1_FDX_s102
                                 + 1.5726 w_k1_IBM_s102
                                 + 0.994139 w_k1_INTC_s102
                                 + 1.15304 w_k1_KO_s102 + 2.22642 w_k1_MAT_s102
                                 + 1.01885 w_k1_MSFT_s102
                                 + 1.02787 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.979444 w_k1_ADM_s102
                                 + 0.613954 w_k1_AXP_s102
                                 + 0.713614 w_k1_DIS_s102
                                 + 0.814785 w_k1_FDX_s102
                                 + 0.609728 w_k1_IBM_s102
                                 + 0.806185 w_k1_INTC_s102
                                 + 0.652893 w_k1_KO_s102
                                 + 0.578949 w_k1_MAT_s102
                                 + 0.823934 w_k1_MSFT_s102
                                 + 0.893461 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.33341 w_k1_ADM_s102
                                 + 1.34603 w_k1_AXP_s102
                                 + 1.51781 w_k1_DIS_s102
                                 + 1.42052 w_k1_FDX_s102
                                 + 0.871644 w_k1_IBM_s102
                                 + 1.23307 w_k1_INTC_s102
                                 + 1.15577 w_k1_KO_s102
                                 + 0.903636 w_k1_MAT_s102
                                 + 1.27216 w_k1_MSFT_s102
                                 + 1.24829 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.12225 w_k1_ADM_s102
                                 + 1.10336 w_k1_AXP_s102
                                 + 1.23645 w_k1_DIS_s102
                                 + 1.18052 w_k1_FDX_s102
                                 + 0.884163 w_k1_IBM_s102
                                 + 1.11572 w_k1_INTC_s102
                                 + 1.06657 w_k1_KO_s102
                                 + 0.928534 w_k1_MAT_s102
                                 + 1.21872 w_k1_MSFT_s102 + 1.1038 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.37642 w_k1_ADM_s102
                                 + 1.19483 w_k1_AXP_s102
                                 + 0.999475 w_k1_DIS_s102
                                 + 0.994918 w_k1_FDX_s102
                                 + 1.15004 w_k1_IBM_s102
                                 + 0.791298 w_k1_INTC_s102
                                 + 1.08554 w_k1_KO_s102 + 1.84805 w_k1_MAT_s102
                                 + 0.934008 w_k1_MSFT_s102
                                 + 1.02662 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.967196 w_k1_ADM_s103
                                 + 0.774853 w_k1_AXP_s103
                                 + 0.926167 w_k1_DIS_s103
                                 + 0.854242 w_k1_FDX_s103
                                 + 1.11949 w_k1_IBM_s103
                                 + 1.05283 w_k1_INTC_s103
                                 + 1.13706 w_k1_KO_s103 + 1.17451 w_k1_MAT_s103
                                 + 1.05149 w_k1_MSFT_s103
                                 + 0.943953 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.17785 w_k1_ADM_s103
                                 + 1.29156 w_k1_AXP_s103
                                 + 1.35045 w_k1_DIS_s103
                                 + 1.20418 w_k1_FDX_s103
                                 + 1.10403 w_k1_IBM_s103
                                 + 1.10561 w_k1_INTC_s103
                                 + 1.19823 w_k1_KO_s103 + 1.34735 w_k1_MAT_s103
                                 + 1.27203 w_k1_MSFT_s103 + 1.1247 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.41078 w_k1_ADM_s103
                                 + 1.45784 w_k1_AXP_s103
                                 + 1.93577 w_k1_DIS_s103
                                 + 1.44683 w_k1_FDX_s103
                                 + 1.15921 w_k1_IBM_s103
                                 + 1.20988 w_k1_INTC_s103 + 1.2026 w_k1_KO_s103
                                 + 1.32077 w_k1_MAT_s103
                                 + 1.33249 w_k1_MSFT_s103
                                 + 1.30412 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.17959 w_k1_ADM_s103
                                 + 1.23376 w_k1_AXP_s103
                                 + 0.886252 w_k1_DIS_s103
                                 + 1.02396 w_k1_FDX_s103
                                 + 0.944729 w_k1_IBM_s103
                                 + 1.09966 w_k1_INTC_s103 + 0.9286 w_k1_KO_s103
                                 + 1.04193 w_k1_MAT_s103
                                 + 1.11512 w_k1_MSFT_s103
                                 + 0.886331 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.25563 w_k1_ADM_s103
                                 + 1.03064 w_k1_AXP_s103
                                 + 1.23117 w_k1_DIS_s103
                                 + 1.18326 w_k1_FDX_s103
                                 + 0.962307 w_k1_IBM_s103
                                 + 1.03893 w_k1_INTC_s103
                                 + 1.04335 w_k1_KO_s103 + 1.14677 w_k1_MAT_s103
                                 + 1.01629 w_k1_MSFT_s103
                                 + 1.03787 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.04264 w_k1_ADM_s103
                                 + 1.16622 w_k1_AXP_s103
                                 + 1.12281 w_k1_DIS_s103
                                 + 0.937305 w_k1_FDX_s103
                                 + 1.53562 w_k1_IBM_s103
                                 + 1.14412 w_k1_INTC_s103
                                 + 1.17765 w_k1_KO_s103 + 1.58005 w_k1_MAT_s103
                                 + 1.14873 w_k1_MSFT_s103 + 1.0672 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.943898 w_k1_ADM_s103
                                 + 0.729075 w_k1_AXP_s103
                                 + 0.619347 w_k1_DIS_s103
                                 + 0.876244 w_k1_FDX_s103
                                 + 0.721824 w_k1_IBM_s103
                                 + 0.805717 w_k1_INTC_s103
                                 + 0.744557 w_k1_KO_s103
                                 + 0.667586 w_k1_MAT_s103
                                 + 0.868437 w_k1_MSFT_s103
                                 + 0.918604 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.30624 w_k1_ADM_s103
                                 + 0.766738 w_k1_AXP_s103
                                 + 1.13503 w_k1_DIS_s103
                                 + 1.27806 w_k1_FDX_s103
                                 + 0.746044 w_k1_IBM_s103
                                 + 1.13716 w_k1_INTC_s103
                                 + 0.988867 w_k1_KO_s103
                                 + 0.894626 w_k1_MAT_s103
                                 + 1.23886 w_k1_MSFT_s103
                                 + 1.05608 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.04352 w_k1_ADM_s104
                                 + 0.916617 w_k1_AXP_s104
                                 + 1.16135 w_k1_DIS_s104
                                 + 1.11245 w_k1_FDX_s104
                                 + 0.835375 w_k1_IBM_s104
                                 + 1.051 w_k1_INTC_s104 + 0.990389 w_k1_KO_s104
                                 + 0.857208 w_k1_MAT_s104
                                 + 1.23543 w_k1_MSFT_s104
                                 + 0.908812 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.775637 w_k1_ADM_s104
                                 + 0.629917 w_k1_AXP_s104
                                 + 0.833085 w_k1_DIS_s104
                                 + 0.809587 w_k1_FDX_s104
                                 + 0.770693 w_k1_IBM_s104
                                 + 0.780291 w_k1_INTC_s104
                                 + 0.745825 w_k1_KO_s104
                                 + 0.692833 w_k1_MAT_s104
                                 + 0.767487 w_k1_MSFT_s104
                                 + 0.807889 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.03205 w_k1_ADM_s104
                                 + 1.01732 w_k1_AXP_s104
                                 + 1.26762 w_k1_DIS_s104 + 0.9931 w_k1_FDX_s104
                                 + 1.00402 w_k1_IBM_s104
                                 + 0.901541 w_k1_INTC_s104
                                 + 1.05374 w_k1_KO_s104 + 1.23357 w_k1_MAT_s104
                                 + 0.950351 w_k1_MSFT_s104
                                 + 0.998225 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.893176 w_k1_ADM_s104
                                 + 0.927278 w_k1_AXP_s104
                                 + 1.00822 w_k1_DIS_s104
                                 + 0.870128 w_k1_FDX_s104
                                 + 1.27842 w_k1_IBM_s104
                                 + 0.946358 w_k1_INTC_s104
                                 + 1.15816 w_k1_KO_s104 + 1.47482 w_k1_MAT_s104
                                 + 1.02185 w_k1_MSFT_s104
                                 + 1.02989 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.25791 w_k1_ADM_s104
                                 + 1.27003 w_k1_AXP_s104
                                 + 1.33726 w_k1_DIS_s104
                                 + 1.24321 w_k1_FDX_s104
                                 + 1.01678 w_k1_IBM_s104
                                 + 1.10831 w_k1_INTC_s104
                                 + 0.917157 w_k1_KO_s104
                                 + 0.927863 w_k1_MAT_s104
                                 + 1.03501 w_k1_MSFT_s104
                                 + 0.95861 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.50693 w_k1_ADM_s104
                                 + 1.85747 w_k1_AXP_s104
                                 + 1.92962 w_k1_DIS_s104
                                 + 1.46286 w_k1_FDX_s104
                                 + 1.42486 w_k1_IBM_s104
                                 + 1.24815 w_k1_INTC_s104
                                 + 1.49328 w_k1_KO_s104 + 1.64802 w_k1_MAT_s104
                                 + 1.28966 w_k1_MSFT_s104
                                 + 1.12441 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.17592 w_k1_ADM_s104
                                 + 1.24704 w_k1_AXP_s104
                                 + 1.61538 w_k1_DIS_s104
                                 + 1.23937 w_k1_FDX_s104
                                 + 1.27426 w_k1_IBM_s104
                                 + 0.979714 w_k1_INTC_s104
                                 + 1.35399 w_k1_KO_s104 + 1.50766 w_k1_MAT_s104
                                 + 1.16948 w_k1_MSFT_s104
                                 + 1.14086 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.12762 w_k1_ADM_s104
                                 + 1.06499 w_k1_AXP_s104
                                 + 1.41704 w_k1_DIS_s104
                                 + 1.09526 w_k1_FDX_s104
                                 + 1.42353 w_k1_IBM_s104
                                 + 0.945963 w_k1_INTC_s104
                                 + 1.52699 w_k1_KO_s104 + 1.46733 w_k1_MAT_s104
                                 + 1.19551 w_k1_MSFT_s104
                                 + 1.08627 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.961368 w_k1_ADM_s105
                                 + 0.75904 w_k1_AXP_s105
                                 + 0.758335 w_k1_DIS_s105
                                 + 0.823418 w_k1_FDX_s105
                                 + 0.643729 w_k1_IBM_s105
                                 + 1.06084 w_k1_INTC_s105
                                 + 0.664633 w_k1_KO_s105
                                 + 0.468415 w_k1_MAT_s105
                                 + 0.915035 w_k1_MSFT_s105
                                 + 0.764521 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.00947 w_k1_ADM_s105
                                 + 1.12634 w_k1_AXP_s105
                                 + 0.959714 w_k1_DIS_s105
                                 + 1.05628 w_k1_FDX_s105
                                 + 0.90602 w_k1_IBM_s105
                                 + 1.17993 w_k1_INTC_s105
                                 + 0.946934 w_k1_KO_s105 + 1.1691 w_k1_MAT_s105
                                 + 1.01034 w_k1_MSFT_s105
                                 + 1.00976 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.11397 w_k1_ADM_s105
                                 + 1.06838 w_k1_AXP_s105
                                 + 0.850542 w_k1_DIS_s105
                                 + 0.866117 w_k1_FDX_s105
                                 + 0.883837 w_k1_IBM_s105
                                 + 1.01312 w_k1_INTC_s105
                                 + 0.813468 w_k1_KO_s105
                                 + 0.905995 w_k1_MAT_s105
                                 + 0.829331 w_k1_MSFT_s105
                                 + 0.91192 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.634992 w_k1_ADM_s105
                                 + 0.624742 w_k1_AXP_s105
                                 + 0.757353 w_k1_DIS_s105
                                 + 0.698868 w_k1_FDX_s105
                                 + 0.634562 w_k1_IBM_s105
                                 + 0.961757 w_k1_INTC_s105
                                 + 0.679176 w_k1_KO_s105
                                 + 0.384122 w_k1_MAT_s105
                                 + 0.782571 w_k1_MSFT_s105
                                 + 0.766761 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.16834 w_k1_ADM_s105
                                 + 1.42934 w_k1_AXP_s105
                                 + 1.20333 w_k1_DIS_s105
                                 + 1.04531 w_k1_FDX_s105
                                 + 1.36089 w_k1_IBM_s105
                                 + 1.07456 w_k1_INTC_s105
                                 + 1.14702 w_k1_KO_s105 + 1.8298 w_k1_MAT_s105
                                 + 0.95207 w_k1_MSFT_s105
                                 + 1.05705 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.50972 w_k1_ADM_s105
                                 + 1.19464 w_k1_AXP_s105
                                 + 1.58275 w_k1_DIS_s105
                                 + 1.25375 w_k1_FDX_s105
                                 + 1.40964 w_k1_IBM_s105
                                 + 1.25492 w_k1_INTC_s105
                                 + 1.10387 w_k1_KO_s105 + 1.20681 w_k1_MAT_s105
                                 + 1.18136 w_k1_MSFT_s105
                                 + 1.11574 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.52397 w_k1_ADM_s105
                                 + 2.03933 w_k1_AXP_s105
                                 + 1.76619 w_k1_DIS_s105 + 1.5291 w_k1_FDX_s105
                                 + 0.826531 w_k1_IBM_s105
                                 + 1.55227 w_k1_INTC_s105
                                 + 1.05934 w_k1_KO_s105 + 1.10928 w_k1_MAT_s105
                                 + 1.34153 w_k1_MSFT_s105
                                 + 1.05334 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.921949 w_k1_ADM_s105
                                 + 0.833198 w_k1_AXP_s105
                                 + 0.950002 w_k1_DIS_s105
                                 + 0.81269 w_k1_FDX_s105
                                 + 1.13517 w_k1_IBM_s105
                                 + 0.843118 w_k1_INTC_s105
                                 + 0.932756 w_k1_KO_s105
                                 + 0.870581 w_k1_MAT_s105
                                 + 0.84542 w_k1_MSFT_s105
                                 + 0.925136 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.822072 w_k1_ADM_s106
                                 + 0.979795 w_k1_AXP_s106
                                 + 1.17547 w_k1_DIS_s106
                                 + 1.12928 w_k1_FDX_s106
                                 + 1.16323 w_k1_IBM_s106
                                 + 1.07427 w_k1_INTC_s106 + 1.0551 w_k1_KO_s106
                                 + 0.88217 w_k1_MAT_s106
                                 + 1.09084 w_k1_MSFT_s106
                                 + 0.936957 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.05975 w_k1_ADM_s106
                                 + 1.08531 w_k1_AXP_s106
                                 + 1.03848 w_k1_DIS_s106
                                 + 0.978794 w_k1_FDX_s106
                                 + 1.43185 w_k1_IBM_s106
                                 + 1.14907 w_k1_INTC_s106
                                 + 1.28233 w_k1_KO_s106 + 1.07522 w_k1_MAT_s106
                                 + 1.15265 w_k1_MSFT_s106
                                 + 1.16017 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.30682 w_k1_ADM_s106
                                 + 0.95399 w_k1_AXP_s106
                                 + 1.06607 w_k1_DIS_s106
                                 + 1.19361 w_k1_FDX_s106
                                 + 0.762661 w_k1_IBM_s106
                                 + 0.931806 w_k1_INTC_s106
                                 + 0.898582 w_k1_KO_s106
                                 + 0.686448 w_k1_MAT_s106
                                 + 1.11603 w_k1_MSFT_s106
                                 + 0.984245 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12976 w_k1_ADM_s106
                                 + 0.900652 w_k1_AXP_s106
                                 + 1.12253 w_k1_DIS_s106
                                 + 1.11413 w_k1_FDX_s106
                                 + 0.942716 w_k1_IBM_s106
                                 + 1.0789 w_k1_INTC_s106 + 1.11654 w_k1_KO_s106
                                 + 0.565124 w_k1_MAT_s106
                                 + 1.11757 w_k1_MSFT_s106
                                 + 1.10313 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.972806 w_k1_ADM_s106
                                 + 1.15079 w_k1_AXP_s106
                                 + 0.811834 w_k1_DIS_s106
                                 + 0.989801 w_k1_FDX_s106
                                 + 1.01179 w_k1_IBM_s106
                                 + 0.901563 w_k1_INTC_s106
                                 + 0.900905 w_k1_KO_s106
                                 + 1.14849 w_k1_MAT_s106
                                 + 1.04666 w_k1_MSFT_s106
                                 + 0.939538 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.735169 w_k1_ADM_s106
                                 + 0.692192 w_k1_AXP_s106
                                 + 0.678443 w_k1_DIS_s106
                                 + 0.820885 w_k1_FDX_s106
                                 + 0.709978 w_k1_IBM_s106
                                 + 0.696469 w_k1_INTC_s106
                                 + 0.783316 w_k1_KO_s106
                                 + 0.681543 w_k1_MAT_s106
                                 + 0.872511 w_k1_MSFT_s106
                                 + 0.892726 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.29648 w_k1_ADM_s106
                                 + 1.26508 w_k1_AXP_s106
                                 + 1.80099 w_k1_DIS_s106
                                 + 1.32475 w_k1_FDX_s106
                                 + 1.32677 w_k1_IBM_s106
                                 + 1.39481 w_k1_INTC_s106
                                 + 1.47044 w_k1_KO_s106
                                 + 0.978117 w_k1_MAT_s106
                                 + 1.48066 w_k1_MSFT_s106
                                 + 1.36045 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.720711 w_k1_ADM_s106
                                 + 0.875367 w_k1_AXP_s106
                                 + 0.78999 w_k1_DIS_s106
                                 + 0.852141 w_k1_FDX_s106
                                 + 0.900448 w_k1_IBM_s106
                                 + 0.932756 w_k1_INTC_s106
                                 + 0.777854 w_k1_KO_s106
                                 + 0.619418 w_k1_MAT_s106
                                 + 0.850756 w_k1_MSFT_s106
                                 + 0.826345 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.01031 w_k1_ADM_s107
                                 + 1.17682 w_k1_AXP_s107
                                 + 0.917867 w_k1_DIS_s107
                                 + 1.09931 w_k1_FDX_s107
                                 + 1.02174 w_k1_IBM_s107
                                 + 0.983738 w_k1_INTC_s107
                                 + 0.926043 w_k1_KO_s107 + 1.1497 w_k1_MAT_s107
                                 + 0.958118 w_k1_MSFT_s107
                                 + 0.99385 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.25686 w_k1_ADM_s107
                                 + 1.64306 w_k1_AXP_s107
                                 + 1.74276 w_k1_DIS_s107
                                 + 1.63807 w_k1_FDX_s107 + 1.1153 w_k1_IBM_s107
                                 + 1.25127 w_k1_INTC_s107
                                 + 1.26196 w_k1_KO_s107 + 1.68641 w_k1_MAT_s107
                                 + 1.29987 w_k1_MSFT_s107
                                 + 1.14663 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.874426 w_k1_ADM_s107
                                 + 1.0518 w_k1_AXP_s107
                                 + 0.799196 w_k1_DIS_s107
                                 + 0.901537 w_k1_FDX_s107
                                 + 1.17543 w_k1_IBM_s107
                                 + 1.12004 w_k1_INTC_s107
                                 + 0.969063 w_k1_KO_s107
                                 + 1.30467 w_k1_MAT_s107
                                 + 0.825059 w_k1_MSFT_s107
                                 + 1.00766 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.941762 w_k1_ADM_s107
                                 + 0.94954 w_k1_AXP_s107
                                 + 0.865464 w_k1_DIS_s107
                                 + 1.00991 w_k1_FDX_s107
                                 + 0.555828 w_k1_IBM_s107
                                 + 1.07239 w_k1_INTC_s107
                                 + 0.721402 w_k1_KO_s107
                                 + 0.500502 w_k1_MAT_s107
                                 + 0.935542 w_k1_MSFT_s107
                                 + 0.871529 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.14022 w_k1_ADM_s107
                                 + 1.12411 w_k1_AXP_s107
                                 + 1.35276 w_k1_DIS_s107
                                 + 1.18216 w_k1_FDX_s107
                                 + 0.762504 w_k1_IBM_s107
                                 + 1.15807 w_k1_INTC_s107
                                 + 0.91699 w_k1_KO_s107
                                 + 0.698991 w_k1_MAT_s107
                                 + 1.21868 w_k1_MSFT_s107
                                 + 1.04766 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.10077 w_k1_ADM_s107
                                 + 0.839845 w_k1_AXP_s107
                                 + 1.2556 w_k1_DIS_s107
                                 + 0.961892 w_k1_FDX_s107
                                 + 1.08558 w_k1_IBM_s107
                                 + 1.00073 w_k1_INTC_s107
                                 + 1.02531 w_k1_KO_s107
                                 + 0.873577 w_k1_MAT_s107
                                 + 1.10225 w_k1_MSFT_s107
                                 + 1.02214 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.15649 w_k1_ADM_s107
                                 + 1.30635 w_k1_AXP_s107
                                 + 1.29877 w_k1_DIS_s107
                                 + 1.19697 w_k1_FDX_s107
                                 + 1.59151 w_k1_IBM_s107
                                 + 1.39402 w_k1_INTC_s107
                                 + 1.46683 w_k1_KO_s107 + 1.29116 w_k1_MAT_s107
                                 + 1.16974 w_k1_MSFT_s107
                                 + 1.17752 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.944042 w_k1_ADM_s107
                                 + 0.657015 w_k1_AXP_s107
                                 + 0.677493 w_k1_DIS_s107
                                 + 0.760112 w_k1_FDX_s107
                                 + 0.784825 w_k1_IBM_s107
                                 + 0.798646 w_k1_INTC_s107
                                 + 0.733183 w_k1_KO_s107
                                 + 0.57387 w_k1_MAT_s107
                                 + 0.799196 w_k1_MSFT_s107
                                 + 0.833387 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.23614 w_k1_ADM_s110
                                 + 1.38403 w_k1_AXP_s110
                                 + 1.53758 w_k1_DIS_s110
                                 + 1.40836 w_k1_FDX_s110
                                 + 1.08344 w_k1_IBM_s110
                                 + 0.934533 w_k1_INTC_s110
                                 + 1.31672 w_k1_KO_s110 + 1.69247 w_k1_MAT_s110
                                 + 1.32359 w_k1_MSFT_s110
                                 + 1.21102 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.771798 w_k1_ADM_s110
                                 + 0.801638 w_k1_AXP_s110
                                 + 0.737768 w_k1_DIS_s110
                                 + 0.759032 w_k1_FDX_s110
                                 + 1.24067 w_k1_IBM_s110
                                 + 0.767352 w_k1_INTC_s110
                                 + 1.05126 w_k1_KO_s110 + 1.54503 w_k1_MAT_s110
                                 + 0.782968 w_k1_MSFT_s110
                                 + 0.999815 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.878959 w_k1_ADM_s110
                                 + 0.762174 w_k1_AXP_s110
                                 + 0.797344 w_k1_DIS_s110
                                 + 0.816958 w_k1_FDX_s110
                                 + 1.00083 w_k1_IBM_s110
                                 + 0.949696 w_k1_INTC_s110
                                 + 0.957685 w_k1_KO_s110
                                 + 1.12152 w_k1_MAT_s110
                                 + 1.02607 w_k1_MSFT_s110
                                 + 0.997624 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.029 w_k1_ADM_s110
                                 + 1.05608 w_k1_AXP_s110
                                 + 0.966484 w_k1_DIS_s110
                                 + 1.04831 w_k1_FDX_s110
                                 + 0.880476 w_k1_IBM_s110
                                 + 0.983224 w_k1_INTC_s110
                                 + 0.978973 w_k1_KO_s110
                                 + 1.24951 w_k1_MAT_s110
                                 + 1.15744 w_k1_MSFT_s110
                                 + 0.990566 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.850352 w_k1_ADM_s110
                                 + 0.716314 w_k1_AXP_s110
                                 + 0.660744 w_k1_DIS_s110
                                 + 0.812595 w_k1_FDX_s110
                                 + 0.700763 w_k1_IBM_s110
                                 + 0.672677 w_k1_INTC_s110
                                 + 0.701416 w_k1_KO_s110
                                 + 1.09324 w_k1_MAT_s110
                                 + 0.842706 w_k1_MSFT_s110
                                 + 0.866768 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.05572 w_k1_ADM_s110
                                 + 1.08508 w_k1_AXP_s110
                                 + 1.30138 w_k1_DIS_s110
                                 + 1.04243 w_k1_FDX_s110
                                 + 1.53524 w_k1_IBM_s110
                                 + 1.01435 w_k1_INTC_s110
                                 + 1.40763 w_k1_KO_s110 + 1.67434 w_k1_MAT_s110
                                 + 1.16702 w_k1_MSFT_s110
                                 + 1.22146 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.08047 w_k1_ADM_s110
                                 + 1.03281 w_k1_AXP_s110
                                 + 1.17915 w_k1_DIS_s110
                                 + 1.16336 w_k1_FDX_s110
                                 + 0.703057 w_k1_IBM_s110
                                 + 0.969116 w_k1_INTC_s110
                                 + 0.910341 w_k1_KO_s110
                                 + 0.880783 w_k1_MAT_s110
                                 + 1.10329 w_k1_MSFT_s110
                                 + 0.941119 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.920282 w_k1_ADM_s110
                                 + 1.06136 w_k1_AXP_s110
                                 + 0.952322 w_k1_DIS_s110
                                 + 0.964423 w_k1_FDX_s110
                                 + 1.05201 w_k1_IBM_s110
                                 + 0.910941 w_k1_INTC_s110
                                 + 1.06608 w_k1_KO_s110 + 1.64636 w_k1_MAT_s110
                                 + 0.928623 w_k1_MSFT_s110
                                 + 0.985201 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.05039 w_k1_ADM_s111
                                 + 1.43895 w_k1_AXP_s111
                                 + 1.59271 w_k1_DIS_s111
                                 + 1.26699 w_k1_FDX_s111
                                 + 1.36142 w_k1_IBM_s111
                                 + 1.49135 w_k1_INTC_s111
                                 + 1.34593 w_k1_KO_s111 + 1.26538 w_k1_MAT_s111
                                 + 1.46449 w_k1_MSFT_s111
                                 + 1.19092 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.03727 w_k1_ADM_s111
                                 + 0.902922 w_k1_AXP_s111
                                 + 1.05893 w_k1_DIS_s111
                                 + 1.02259 w_k1_FDX_s111
                                 + 0.68828 w_k1_IBM_s111
                                 + 1.09456 w_k1_INTC_s111
                                 + 0.917203 w_k1_KO_s111
                                 + 0.498362 w_k1_MAT_s111
                                 + 1.12886 w_k1_MSFT_s111
                                 + 0.835599 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.820983 w_k1_ADM_s111
                                 + 1.01606 w_k1_AXP_s111
                                 + 0.777882 w_k1_DIS_s111
                                 + 0.831589 w_k1_FDX_s111
                                 + 0.912407 w_k1_IBM_s111
                                 + 1.04041 w_k1_INTC_s111
                                 + 0.861943 w_k1_KO_s111
                                 + 0.699966 w_k1_MAT_s111
                                 + 0.839195 w_k1_MSFT_s111
                                 + 0.84746 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.02768 w_k1_ADM_s111
                                 + 1.20323 w_k1_AXP_s111
                                 + 0.96152 w_k1_DIS_s111
                                 + 1.18177 w_k1_FDX_s111
                                 + 0.912111 w_k1_IBM_s111
                                 + 1.01338 w_k1_INTC_s111
                                 + 0.938212 w_k1_KO_s111
                                 + 0.861485 w_k1_MAT_s111
                                 + 1.04146 w_k1_MSFT_s111
                                 + 0.970089 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.8121 w_k1_ADM_s111
                                 + 1.01116 w_k1_AXP_s111
                                 + 0.844416 w_k1_DIS_s111
                                 + 0.883178 w_k1_FDX_s111
                                 + 1.46709 w_k1_IBM_s111
                                 + 1.02288 w_k1_INTC_s111
                                 + 1.01348 w_k1_KO_s111 + 1.17677 w_k1_MAT_s111
                                 + 0.914768 w_k1_MSFT_s111
                                 + 0.913484 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.988175 w_k1_ADM_s111
                                 + 0.994169 w_k1_AXP_s111
                                 + 1.10672 w_k1_DIS_s111
                                 + 0.965183 w_k1_FDX_s111
                                 + 1.16928 w_k1_IBM_s111
                                 + 0.926131 w_k1_INTC_s111
                                 + 1.02331 w_k1_KO_s111 + 1.15549 w_k1_MAT_s111
                                 + 1.11901 w_k1_MSFT_s111
                                 + 1.00962 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.811681 w_k1_ADM_s111
                                 + 0.724195 w_k1_AXP_s111
                                 + 0.684423 w_k1_DIS_s111
                                 + 0.771982 w_k1_FDX_s111
                                 + 0.718802 w_k1_IBM_s111
                                 + 0.853067 w_k1_INTC_s111
                                 + 0.665433 w_k1_KO_s111
                                 + 0.597229 w_k1_MAT_s111
                                 + 0.733265 w_k1_MSFT_s111
                                 + 0.761748 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.87017 w_k1_ADM_s111
                                 + 0.606522 w_k1_AXP_s111
                                 + 0.671189 w_k1_DIS_s111
                                 + 0.784027 w_k1_FDX_s111
                                 + 0.997074 w_k1_IBM_s111
                                 + 0.882092 w_k1_INTC_s111
                                 + 0.824573 w_k1_KO_s111
                                 + 0.656561 w_k1_MAT_s111
                                 + 0.921954 w_k1_MSFT_s111
                                 + 0.777995 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.16382 w_k1_ADM_s112
                                 + 1.55469 w_k1_AXP_s112
                                 + 1.73753 w_k1_DIS_s112
                                 + 1.36532 w_k1_FDX_s112
                                 + 1.49427 w_k1_IBM_s112
                                 + 1.39487 w_k1_INTC_s112
                                 + 1.61163 w_k1_KO_s112 + 1.1913 w_k1_MAT_s112
                                 + 1.31649 w_k1_MSFT_s112
                                 + 1.36194 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.712028 w_k1_ADM_s112
                                 + 0.974573 w_k1_AXP_s112
                                 + 0.804455 w_k1_DIS_s112
                                 + 0.824288 w_k1_FDX_s112
                                 + 0.986584 w_k1_IBM_s112
                                 + 0.813337 w_k1_INTC_s112
                                 + 0.871901 w_k1_KO_s112
                                 + 1.12301 w_k1_MAT_s112
                                 + 0.787704 w_k1_MSFT_s112
                                 + 0.924585 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.66667 w_k1_ADM_s112
                                 + 0.872338 w_k1_AXP_s112
                                 + 1.0412 w_k1_DIS_s112
                                 + 0.918024 w_k1_FDX_s112
                                 + 1.1333 w_k1_IBM_s112
                                 + 1.07791 w_k1_INTC_s112 + 1.1098 w_k1_KO_s112
                                 + 0.998867 w_k1_MAT_s112
                                 + 0.997298 w_k1_MSFT_s112
                                 + 0.971861 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.902193 w_k1_ADM_s112
                                 + 1.27439 w_k1_AXP_s112
                                 + 0.945694 w_k1_DIS_s112
                                 + 0.956952 w_k1_FDX_s112
                                 + 1.27084 w_k1_IBM_s112
                                 + 1.16167 w_k1_INTC_s112
                                 + 1.05534 w_k1_KO_s112 + 1.33921 w_k1_MAT_s112
                                 + 0.833992 w_k1_MSFT_s112
                                 + 1.01848 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.929627 w_k1_ADM_s112
                                 + 1.1776 w_k1_AXP_s112 + 1.02119 w_k1_DIS_s112
                                 + 1.09716 w_k1_FDX_s112
                                 + 1.08324 w_k1_IBM_s112
                                 + 1.21952 w_k1_INTC_s112
                                 + 1.00888 w_k1_KO_s112
                                 + 0.832735 w_k1_MAT_s112
                                 + 0.996542 w_k1_MSFT_s112
                                 + 0.971504 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.720996 w_k1_ADM_s112
                                 + 0.758057 w_k1_AXP_s112
                                 + 0.663865 w_k1_DIS_s112
                                 + 0.783847 w_k1_FDX_s112
                                 + 0.734989 w_k1_IBM_s112
                                 + 0.977719 w_k1_INTC_s112
                                 + 0.73773 w_k1_KO_s112
                                 + 0.717021 w_k1_MAT_s112
                                 + 0.841705 w_k1_MSFT_s112
                                 + 0.87964 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.92608 w_k1_ADM_s112
                                 + 1.17634 w_k1_AXP_s112
                                 + 1.23144 w_k1_DIS_s112
                                 + 1.20149 w_k1_FDX_s112
                                 + 0.848512 w_k1_IBM_s112
                                 + 1.33465 w_k1_INTC_s112
                                 + 1.07109 w_k1_KO_s112 + 0.87404 w_k1_MAT_s112
                                 + 1.11605 w_k1_MSFT_s112
                                 + 1.11958 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.00028 w_k1_ADM_s112
                                 + 1.62796 w_k1_AXP_s112
                                 + 1.17214 w_k1_DIS_s112
                                 + 1.16114 w_k1_FDX_s112
                                 + 1.51757 w_k1_IBM_s112
                                 + 1.37309 w_k1_INTC_s112
                                 + 1.21573 w_k1_KO_s112 + 1.66919 w_k1_MAT_s112
                                 + 1.16077 w_k1_MSFT_s112
                                 + 1.15531 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.22683 w_k1_ADM_s113
                                 + 1.15602 w_k1_AXP_s113
                                 + 0.961299 w_k1_DIS_s113
                                 + 0.973296 w_k1_FDX_s113
                                 + 1.57744 w_k1_IBM_s113
                                 + 1.16073 w_k1_INTC_s113
                                 + 1.34451 w_k1_KO_s113 + 1.60325 w_k1_MAT_s113
                                 + 0.949055 w_k1_MSFT_s113
                                 + 1.09108 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.903234 w_k1_ADM_s113
                                 + 0.73118 w_k1_AXP_s113
                                 + 0.801641 w_k1_DIS_s113
                                 + 0.875352 w_k1_FDX_s113
                                 + 0.92596 w_k1_IBM_s113
                                 + 0.876299 w_k1_INTC_s113
                                 + 1.15319 w_k1_KO_s113
                                 + 0.914587 w_k1_MAT_s113
                                 + 0.821048 w_k1_MSFT_s113
                                 + 0.904193 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.13016 w_k1_ADM_s113
                                 + 1.09212 w_k1_AXP_s113
                                 + 1.52464 w_k1_DIS_s113 + 1.2067 w_k1_FDX_s113
                                 + 1.22363 w_k1_IBM_s113
                                 + 1.00585 w_k1_INTC_s113
                                 + 1.37522 w_k1_KO_s113 + 1.37734 w_k1_MAT_s113
                                 + 1.17681 w_k1_MSFT_s113
                                 + 1.28791 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.901663 w_k1_ADM_s113
                                 + 0.868339 w_k1_AXP_s113
                                 + 0.779992 w_k1_DIS_s113
                                 + 0.694761 w_k1_FDX_s113
                                 + 1.29214 w_k1_IBM_s113
                                 + 0.736247 w_k1_INTC_s113
                                 + 1.01846 w_k1_KO_s113 + 1.09775 w_k1_MAT_s113
                                 + 0.697789 w_k1_MSFT_s113
                                 + 0.979471 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.18726 w_k1_ADM_s113
                                 + 1.15456 w_k1_AXP_s113
                                 + 1.36706 w_k1_DIS_s113
                                 + 1.32863 w_k1_FDX_s113
                                 + 0.783248 w_k1_IBM_s113
                                 + 1.04405 w_k1_INTC_s113
                                 + 1.17331 w_k1_KO_s113 + 1.23657 w_k1_MAT_s113
                                 + 1.20162 w_k1_MSFT_s113
                                 + 1.08527 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.25561 w_k1_ADM_s113
                                 + 1.19795 w_k1_AXP_s113
                                 + 1.02727 w_k1_DIS_s113
                                 + 1.16601 w_k1_FDX_s113
                                 + 1.04636 w_k1_IBM_s113
                                 + 0.985606 w_k1_INTC_s113
                                 + 1.13219 w_k1_KO_s113 + 1.35389 w_k1_MAT_s113
                                 + 1.05828 w_k1_MSFT_s113
                                 + 1.02427 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.15236 w_k1_ADM_s113
                                 + 0.854571 w_k1_AXP_s113
                                 + 0.923958 w_k1_DIS_s113
                                 + 1.02848 w_k1_FDX_s113
                                 + 0.569219 w_k1_IBM_s113
                                 + 0.818904 w_k1_INTC_s113
                                 + 0.779232 w_k1_KO_s113
                                 + 0.360814 w_k1_MAT_s113
                                 + 0.822639 w_k1_MSFT_s113
                                 + 0.95034 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.1816 w_k1_ADM_s113
                                 + 1.23974 w_k1_AXP_s113
                                 + 1.61824 w_k1_DIS_s113
                                 + 1.14704 w_k1_FDX_s113
                                 + 1.77818 w_k1_IBM_s113
                                 + 1.13119 w_k1_INTC_s113
                                 + 1.54227 w_k1_KO_s113 + 2.20874 w_k1_MAT_s113
                                 + 1.17527 w_k1_MSFT_s113
                                 + 1.12201 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.21177 w_k1_ADM_s114
                                 + 1.05982 w_k1_AXP_s114
                                 + 1.20027 w_k1_DIS_s114
                                 + 1.19814 w_k1_FDX_s114
                                 + 1.05678 w_k1_IBM_s114
                                 + 1.32191 w_k1_INTC_s114
                                 + 1.02894 w_k1_KO_s114 + 1.15859 w_k1_MAT_s114
                                 + 1.26363 w_k1_MSFT_s114
                                 + 1.01683 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.873974 w_k1_ADM_s114
                                 + 0.641996 w_k1_AXP_s114
                                 + 0.606664 w_k1_DIS_s114
                                 + 0.748988 w_k1_FDX_s114
                                 + 0.638898 w_k1_IBM_s114
                                 + 0.793284 w_k1_INTC_s114
                                 + 0.699672 w_k1_KO_s114
                                 + 0.611315 w_k1_MAT_s114
                                 + 0.828888 w_k1_MSFT_s114
                                 + 0.803671 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.28961 w_k1_ADM_s114
                                 + 1.82644 w_k1_AXP_s114 + 1.8389 w_k1_DIS_s114
                                 + 1.56497 w_k1_FDX_s114
                                 + 1.28837 w_k1_IBM_s114
                                 + 1.40754 w_k1_INTC_s114
                                 + 1.33457 w_k1_KO_s114 + 1.3917 w_k1_MAT_s114
                                 + 1.45076 w_k1_MSFT_s114
                                 + 1.34876 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.908583 w_k1_ADM_s114
                                 + 0.723776 w_k1_AXP_s114
                                 + 0.600313 w_k1_DIS_s114
                                 + 0.654017 w_k1_FDX_s114
                                 + 1.10425 w_k1_IBM_s114
                                 + 0.710115 w_k1_INTC_s114
                                 + 0.805406 w_k1_KO_s114
                                 + 1.20806 w_k1_MAT_s114
                                 + 0.717006 w_k1_MSFT_s114
                                 + 0.805174 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.81695 w_k1_ADM_s114
                                 + 0.809995 w_k1_AXP_s114
                                 + 0.594446 w_k1_DIS_s114
                                 + 0.807554 w_k1_FDX_s114
                                 + 0.861156 w_k1_IBM_s114
                                 + 0.85235 w_k1_INTC_s114
                                 + 0.66865 w_k1_KO_s114 + 1.08651 w_k1_MAT_s114
                                 + 0.761857 w_k1_MSFT_s114
                                 + 0.755381 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.994747 w_k1_ADM_s114
                                 + 1.14099 w_k1_AXP_s114
                                 + 0.936374 w_k1_DIS_s114
                                 + 1.07061 w_k1_FDX_s114
                                 + 0.757546 w_k1_IBM_s114
                                 + 1.08404 w_k1_INTC_s114
                                 + 0.952653 w_k1_KO_s114
                                 + 1.06727 w_k1_MAT_s114
                                 + 1.22241 w_k1_MSFT_s114
                                 + 0.98882 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.02395 w_k1_ADM_s114
                                 + 0.912901 w_k1_AXP_s114
                                 + 0.821152 w_k1_DIS_s114
                                 + 0.97365 w_k1_FDX_s114
                                 + 0.992018 w_k1_IBM_s114
                                 + 0.959083 w_k1_INTC_s114
                                 + 1.00562 w_k1_KO_s114 + 1.12857 w_k1_MAT_s114
                                 + 1.07436 w_k1_MSFT_s114
                                 + 0.922217 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.09214 w_k1_ADM_s114
                                 + 1.10593 w_k1_AXP_s114
                                 + 1.22644 w_k1_DIS_s114
                                 + 0.951996 w_k1_FDX_s114
                                 + 1.51321 w_k1_IBM_s114
                                 + 1.10834 w_k1_INTC_s114
                                 + 1.33525 w_k1_KO_s114 + 1.85939 w_k1_MAT_s114
                                 + 1.07075 w_k1_MSFT_s114
                                 + 1.06999 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.780251 w_k1_ADM_s115
                                 + 0.746552 w_k1_AXP_s115
                                 + 0.753547 w_k1_DIS_s115
                                 + 0.807191 w_k1_FDX_s115
                                 + 1.2047 w_k1_IBM_s115
                                 + 0.950887 w_k1_INTC_s115
                                 + 0.939435 w_k1_KO_s115
                                 + 0.940503 w_k1_MAT_s115
                                 + 0.831706 w_k1_MSFT_s115
                                 + 0.881767 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.773858 w_k1_ADM_s115
                                 + 0.77186 w_k1_AXP_s115
                                 + 0.900905 w_k1_DIS_s115
                                 + 0.978033 w_k1_FDX_s115
                                 + 0.702428 w_k1_IBM_s115
                                 + 1.04095 w_k1_INTC_s115
                                 + 0.805306 w_k1_KO_s115
                                 + 0.668709 w_k1_MAT_s115
                                 + 0.986934 w_k1_MSFT_s115
                                 + 0.917074 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.892236 w_k1_ADM_s115
                                 + 0.770975 w_k1_AXP_s115
                                 + 1.01866 w_k1_DIS_s115
                                 + 0.94087 w_k1_FDX_s115
                                 + 1.27363 w_k1_IBM_s115
                                 + 0.946671 w_k1_INTC_s115
                                 + 1.04414 w_k1_KO_s115 + 1.34013 w_k1_MAT_s115
                                 + 1.01691 w_k1_MSFT_s115
                                 + 1.07087 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.776152 w_k1_ADM_s115
                                 + 1.05892 w_k1_AXP_s115
                                 + 1.08479 w_k1_DIS_s115
                                 + 0.908808 w_k1_FDX_s115
                                 + 1.57483 w_k1_IBM_s115
                                 + 0.966344 w_k1_INTC_s115
                                 + 1.2746 w_k1_KO_s115 + 1.77525 w_k1_MAT_s115
                                 + 1.0602 w_k1_MSFT_s115 + 1.20377 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.944344 w_k1_ADM_s115
                                 + 1.0653 w_k1_AXP_s115 + 1.29045 w_k1_DIS_s115
                                 + 1.21625 w_k1_FDX_s115
                                 + 0.989798 w_k1_IBM_s115
                                 + 1.33695 w_k1_INTC_s115
                                 + 1.03831 w_k1_KO_s115 + 1.06072 w_k1_MAT_s115
                                 + 1.2924 w_k1_MSFT_s115 + 1.06341 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.09379 w_k1_ADM_s115
                                 + 0.962564 w_k1_AXP_s115
                                 + 0.780783 w_k1_DIS_s115
                                 + 1.0088 w_k1_FDX_s115
                                 + 0.897245 w_k1_IBM_s115
                                 + 0.880841 w_k1_INTC_s115
                                 + 0.802636 w_k1_KO_s115
                                 + 0.973421 w_k1_MAT_s115
                                 + 0.936891 w_k1_MSFT_s115
                                 + 1.01279 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.1442 w_k1_ADM_s115
                                 + 1.25235 w_k1_AXP_s115
                                 + 1.29113 w_k1_DIS_s115
                                 + 1.18835 w_k1_FDX_s115
                                 + 1.26769 w_k1_IBM_s115
                                 + 1.11253 w_k1_INTC_s115
                                 + 1.20577 w_k1_KO_s115 + 1.31877 w_k1_MAT_s115
                                 + 1.14489 w_k1_MSFT_s115
                                 + 1.11977 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.811396 w_k1_ADM_s115
                                 + 0.642749 w_k1_AXP_s115
                                 + 0.610126 w_k1_DIS_s115
                                 + 0.730834 w_k1_FDX_s115
                                 + 0.806991 w_k1_IBM_s115
                                 + 0.747948 w_k1_INTC_s115
                                 + 0.759624 w_k1_KO_s115
                                 + 0.831764 w_k1_MAT_s115
                                 + 0.774608 w_k1_MSFT_s115
                                 + 0.828159 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.983789 w_k1_ADM_s116
                                 + 1.02231 w_k1_AXP_s116 + 1.2227 w_k1_DIS_s116
                                 + 0.974207 w_k1_FDX_s116
                                 + 1.51063 w_k1_IBM_s116
                                 + 1.2424 w_k1_INTC_s116 + 1.18168 w_k1_KO_s116
                                 + 1.2439 w_k1_MAT_s116
                                 + 0.951182 w_k1_MSFT_s116
                                 + 1.06319 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.974004 w_k1_ADM_s116
                                 + 1.38387 w_k1_AXP_s116
                                 + 1.07995 w_k1_DIS_s116
                                 + 1.08243 w_k1_FDX_s116
                                 + 1.19617 w_k1_IBM_s116
                                 + 1.08653 w_k1_INTC_s116
                                 + 0.929682 w_k1_KO_s116
                                 + 0.871357 w_k1_MAT_s116
                                 + 0.980972 w_k1_MSFT_s116
                                 + 0.973581 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.22412 w_k1_ADM_s116
                                 + 0.861245 w_k1_AXP_s116
                                 + 1.16738 w_k1_DIS_s116
                                 + 1.11731 w_k1_FDX_s116
                                 + 0.912106 w_k1_IBM_s116
                                 + 1.12153 w_k1_INTC_s116
                                 + 1.13517 w_k1_KO_s116
                                 + 0.625663 w_k1_MAT_s116
                                 + 1.27072 w_k1_MSFT_s116
                                 + 1.05138 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.82869 w_k1_ADM_s116
                                 + 0.583223 w_k1_AXP_s116
                                 + 0.695652 w_k1_DIS_s116
                                 + 0.75385 w_k1_FDX_s116
                                 + 0.725823 w_k1_IBM_s116
                                 + 1.02101 w_k1_INTC_s116
                                 + 0.70567 w_k1_KO_s116
                                 + 0.423155 w_k1_MAT_s116
                                 + 0.806037 w_k1_MSFT_s116
                                 + 0.835256 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.8443 w_k1_ADM_s116
                                 + 1.03582 w_k1_AXP_s116
                                 + 0.733389 w_k1_DIS_s116
                                 + 0.855411 w_k1_FDX_s116
                                 + 0.96138 w_k1_IBM_s116
                                 + 0.933215 w_k1_INTC_s116
                                 + 0.706168 w_k1_KO_s116
                                 + 0.951401 w_k1_MAT_s116
                                 + 0.740753 w_k1_MSFT_s116
                                 + 0.803254 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.03451 w_k1_ADM_s116
                                 + 1.04612 w_k1_AXP_s116
                                 + 0.898869 w_k1_DIS_s116
                                 + 1.05491 w_k1_FDX_s116
                                 + 0.74341 w_k1_IBM_s116
                                 + 1.16016 w_k1_INTC_s116
                                 + 0.833751 w_k1_KO_s116
                                 + 0.676081 w_k1_MAT_s116
                                 + 1.00324 w_k1_MSFT_s116
                                 + 1.02051 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.16191 w_k1_ADM_s116
                                 + 1.05897 w_k1_AXP_s116
                                 + 1.59161 w_k1_DIS_s116
                                 + 1.27519 w_k1_FDX_s116
                                 + 1.14066 w_k1_IBM_s116
                                 + 1.20203 w_k1_INTC_s116
                                 + 1.15694 w_k1_KO_s116
                                 + 0.785567 w_k1_MAT_s116
                                 + 1.37465 w_k1_MSFT_s116
                                 + 1.15335 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.764303 w_k1_ADM_s116
                                 + 0.803648 w_k1_AXP_s116
                                 + 0.86395 w_k1_DIS_s116
                                 + 0.810482 w_k1_FDX_s116
                                 + 1.14674 w_k1_IBM_s116
                                 + 1.06218 w_k1_INTC_s116
                                 + 0.988123 w_k1_KO_s116
                                 + 0.781776 w_k1_MAT_s116
                                 + 0.94883 w_k1_MSFT_s116
                                 + 0.898085 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.985776 w_k1_ADM_s117
                                 + 0.930761 w_k1_AXP_s117
                                 + 0.64369 w_k1_DIS_s117
                                 + 0.875126 w_k1_FDX_s117
                                 + 0.719803 w_k1_IBM_s117
                                 + 0.922181 w_k1_INTC_s117
                                 + 0.666372 w_k1_KO_s117
                                 + 0.743928 w_k1_MAT_s117
                                 + 0.83638 w_k1_MSFT_s117
                                 + 1.00538 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.707227 w_k1_ADM_s117
                                 + 0.687495 w_k1_AXP_s117
                                 + 0.546432 w_k1_DIS_s117
                                 + 0.776901 w_k1_FDX_s117
                                 + 0.915011 w_k1_IBM_s117
                                 + 0.84362 w_k1_INTC_s117
                                 + 0.771828 w_k1_KO_s117
                                 + 0.745765 w_k1_MAT_s117
                                 + 0.67361 w_k1_MSFT_s117
                                 + 0.861161 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.858738 w_k1_ADM_s117
                                 + 1.02418 w_k1_AXP_s117
                                 + 0.838636 w_k1_DIS_s117
                                 + 1.02043 w_k1_FDX_s117
                                 + 0.85957 w_k1_IBM_s117
                                 + 1.03366 w_k1_INTC_s117
                                 + 0.864746 w_k1_KO_s117
                                 + 0.732228 w_k1_MAT_s117
                                 + 1.03115 w_k1_MSFT_s117
                                 + 1.08697 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.07275 w_k1_ADM_s117
                                 + 1.17465 w_k1_AXP_s117
                                 + 1.14182 w_k1_DIS_s117
                                 + 1.12772 w_k1_FDX_s117
                                 + 1.43754 w_k1_IBM_s117
                                 + 0.977514 w_k1_INTC_s117
                                 + 1.32068 w_k1_KO_s117 + 1.37275 w_k1_MAT_s117
                                 + 1.03741 w_k1_MSFT_s117
                                 + 1.24302 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.01138 w_k1_ADM_s117
                                 + 1.10947 w_k1_AXP_s117
                                 + 1.34103 w_k1_DIS_s117
                                 + 1.09225 w_k1_FDX_s117
                                 + 1.11606 w_k1_IBM_s117
                                 + 0.993669 w_k1_INTC_s117
                                 + 1.00389 w_k1_KO_s117 + 1.03713 w_k1_MAT_s117
                                 + 0.991545 w_k1_MSFT_s117
                                 + 1.07119 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.942576 w_k1_ADM_s117
                                 + 1.08176 w_k1_AXP_s117
                                 + 0.767053 w_k1_DIS_s117
                                 + 0.804378 w_k1_FDX_s117
                                 + 1.31411 w_k1_IBM_s117
                                 + 0.845524 w_k1_INTC_s117
                                 + 1.01475 w_k1_KO_s117 + 1.11854 w_k1_MAT_s117
                                 + 0.833081 w_k1_MSFT_s117
                                 + 0.953557 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.18755 w_k1_ADM_s117
                                 + 1.35639 w_k1_AXP_s117
                                 + 1.14296 w_k1_DIS_s117
                                 + 1.21048 w_k1_FDX_s117
                                 + 1.13415 w_k1_IBM_s117
                                 + 1.13665 w_k1_INTC_s117
                                 + 0.98865 w_k1_KO_s117 + 1.026 w_k1_MAT_s117
                                 + 1.04266 w_k1_MSFT_s117
                                 + 1.10156 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.957792 w_k1_ADM_s117
                                 + 1.06344 w_k1_AXP_s117
                                 + 1.22256 w_k1_DIS_s117
                                 + 1.28685 w_k1_FDX_s117
                                 + 0.707449 w_k1_IBM_s117
                                 + 1.11776 w_k1_INTC_s117
                                 + 1.00349 w_k1_KO_s117
                                 + 0.851861 w_k1_MAT_s117
                                 + 1.21463 w_k1_MSFT_s117
                                 + 1.13882 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0280897657281839 w_k1_ADM_s000
                                 + 0.015659976572803 w_k1_ADM_s001
                                 + 0.0487139311029941 w_k1_ADM_s002
                                 + 0.0229614101671786 w_k1_ADM_s003
                                 + 0.0776791857924437 w_k1_ADM_s004
                                 + 0.0309453215367364 w_k1_ADM_s005
                                 + 0.0552231495697959 w_k1_ADM_s006
                                 + 0.0264549652342957 w_k1_ADM_s007
                                 + 0.0251660440015792 w_k1_ADM_s010
                                 + 0.0355850903645705 w_k1_ADM_s011
                                 + 0.0440551256300588 w_k1_ADM_s012
                                 + 0.0102170833305698 w_k1_ADM_s013
                                 + 0.014978771288854 w_k1_ADM_s014
                                 + 0.0296293893349857 w_k1_ADM_s015
                                 + 0.0109572336969867 w_k1_ADM_s016
                                 + 0.0201081841442106 w_k1_ADM_s017
                                 + 0.0125306740388193 w_k1_ADM_s100
                                 + 0.0337313904548571 w_k1_ADM_s101
                                 + 0.00974129346929908 w_k1_ADM_s102
                                 + 0.0102214261916262 w_k1_ADM_s103
                                 + 0.0278693178544414 w_k1_ADM_s104
                                 + 0.0157200884144276 w_k1_ADM_s105
                                 + 0.00751444400198407 w_k1_ADM_s106
                                 + 0.010962653202075 w_k1_ADM_s107
                                 + 0.0546109042040486 w_k1_ADM_s110
                                 + 0.0375078852661454 w_k1_ADM_s111
                                 + 0.0453086678949992 w_k1_ADM_s112
                                 + 0.0728113128290348 w_k1_ADM_s113
                                 + 0.0451198325439243 w_k1_ADM_s114
                                 + 0.0464879396077897 w_k1_ADM_s115
                                 + 0.0289440763317874 w_k1_ADM_s116
                                 + 0.0348718592857087 w_k1_ADM_s117
                                 + 0.0258473484207046 w_k1_AXP_s000
                                 + 0.0160044383430944 w_k1_AXP_s001
                                 + 0.0501961485871865 w_k1_AXP_s002
                                 + 0.0270050205824039 w_k1_AXP_s003
                                 + 0.0765332841535601 w_k1_AXP_s004
                                 + 0.0268493700836112 w_k1_AXP_s005
                                 + 0.0584862002583942 w_k1_AXP_s006
                                 + 0.0261393388198771 w_k1_AXP_s007
                                 + 0.0255587218357412 w_k1_AXP_s010
                                 + 0.0304191461786042 w_k1_AXP_s011
                                 + 0.0536910054846658 w_k1_AXP_s012
                                 + 0.00984670636160027 w_k1_AXP_s013
                                 + 0.0174441533085151 w_k1_AXP_s014
                                 + 0.0360224658446651 w_k1_AXP_s015
                                 + 0.0133269086978701 w_k1_AXP_s016
                                 + 0.0189837956309034 w_k1_AXP_s017
                                 + 0.0132113513648436 w_k1_AXP_s100
                                 + 0.0414924253277021 w_k1_AXP_s101
                                 + 0.00940699625341952 w_k1_AXP_s102
                                 + 0.00912896658234386 w_k1_AXP_s103
                                 + 0.027165760664868 w_k1_AXP_s104
                                 + 0.0157599522286003 w_k1_AXP_s105
                                 + 0.00790180698041106 w_k1_AXP_s106
                                 + 0.0109268016674245 w_k1_AXP_s107
                                 + 0.0559699890067468 w_k1_AXP_s110
                                 + 0.0420665680675188 w_k1_AXP_s111
                                 + 0.0590423519302896 w_k1_AXP_s112
                                 + 0.0655224327587932 w_k1_AXP_s113
                                 + 0.0453949591642797 w_k1_AXP_s114
                                 + 0.0476426200679299 w_k1_AXP_s115
                                 + 0.0280172790179175 w_k1_AXP_s116
                                 + 0.0381285645267413 w_k1_AXP_s117
                                 + 0.0293042643874036 w_k1_DIS_s000
                                 + 0.0180686026331519 w_k1_DIS_s001
                                 + 0.0504107695908678 w_k1_DIS_s002
                                 + 0.0246247242046167 w_k1_DIS_s003
                                 + 0.0738871086691698 w_k1_DIS_s004
                                 + 0.0323443585409843 w_k1_DIS_s005
                                 + 0.0535874538656783 w_k1_DIS_s006
                                 + 0.0264537598888148 w_k1_DIS_s007
                                 + 0.0248201419694849 w_k1_DIS_s010
                                 + 0.031632495354569 w_k1_DIS_s011
                                 + 0.0491406688446864 w_k1_DIS_s012
                                 + 0.0114469517052953 w_k1_DIS_s013
                                 + 0.0170184696996321 w_k1_DIS_s014
                                 + 0.0396524057941083 w_k1_DIS_s015
                                 + 0.0138730852694839 w_k1_DIS_s016
                                 + 0.0202349622650792 w_k1_DIS_s017
                                 + 0.0132524065885429 w_k1_DIS_s100
                                 + 0.0395258221213506 w_k1_DIS_s101
                                 + 0.0091974714808157 w_k1_DIS_s102
                                 + 0.00949592338773648 w_k1_DIS_s103
                                 + 0.0323815384025381 w_k1_DIS_s104
                                 + 0.0153923569953234 w_k1_DIS_s105
                                 + 0.00791413035873402 w_k1_DIS_s106
                                 + 0.0114065205419251 w_k1_DIS_s107
                                 + 0.0568665551431035 w_k1_DIS_s110
                                 + 0.0414634626740564 w_k1_DIS_s111
                                 + 0.0551978375555967 w_k1_DIS_s112
                                 + 0.0709605059152729 w_k1_DIS_s113
                                 + 0.0437603332010275 w_k1_DIS_s114
                                 + 0.052035105262268 w_k1_DIS_s115
                                 + 0.0318767976536699 w_k1_DIS_s116
                                 + 0.0357945020654134 w_k1_DIS_s117
                                 + 0.0277205574782773 w_k1_FDX_s000
                                 + 0.0160466313757372 w_k1_FDX_s001
                                 + 0.0460656477340003 w_k1_FDX_s002
                                 + 0.0245345380301777 w_k1_FDX_s003
                                 + 0.0734892494447606 w_k1_FDX_s004
                                 + 0.0316124928836608 w_k1_FDX_s005
                                 + 0.0538462758082374 w_k1_FDX_s006
                                 + 0.0273514085450658 w_k1_FDX_s007
                                 + 0.0251888366430279 w_k1_FDX_s010
                                 + 0.0320505147066592 w_k1_FDX_s011
                                 + 0.0491726521055976 w_k1_FDX_s012
                                 + 0.0104967369879245 w_k1_FDX_s013
                                 + 0.017847139631515 w_k1_FDX_s014
                                 + 0.0378331012582215 w_k1_FDX_s015
                                 + 0.0123632764590584 w_k1_FDX_s016
                                 + 0.0196855036560128 w_k1_FDX_s017
                                 + 0.0120871069531935 w_k1_FDX_s100
                                 + 0.035993988967288 w_k1_FDX_s101
                                 + 0.00889746585825979 w_k1_FDX_s102
                                 + 0.00946491035405704 w_k1_FDX_s103
                                 + 0.028109461744277 w_k1_FDX_s104
                                 + 0.0142234738852045 w_k1_FDX_s105
                                 + 0.00787604956092583 w_k1_FDX_s106
                                 + 0.0110545087104492 w_k1_FDX_s107
                                 + 0.0560249178594744 w_k1_FDX_s110
                                 + 0.0399531999765493 w_k1_FDX_s111
                                 + 0.0539729684671601 w_k1_FDX_s112
                                 + 0.0693778732915237 w_k1_FDX_s113
                                 + 0.0443533866932253 w_k1_FDX_s114
                                 + 0.0524062938066968 w_k1_FDX_s115
                                 + 0.0293221855444911 w_k1_FDX_s116
                                 + 0.0376352136590517 w_k1_FDX_s117
                                 + 0.0277867168696749 w_k1_IBM_s000
                                 + 0.0161226319079473 w_k1_IBM_s001
                                 + 0.0482243639661614 w_k1_IBM_s002
                                 + 0.0246770957862814 w_k1_IBM_s003
                                 + 0.0765879293395374 w_k1_IBM_s004
                                 + 0.0321076640819707 w_k1_IBM_s005
                                 + 0.0522846599136357 w_k1_IBM_s006
                                 + 0.0254058403608852 w_k1_IBM_s007
                                 + 0.0245827807515463 w_k1_IBM_s010
                                 + 0.033272579791791 w_k1_IBM_s011
                                 + 0.0491897815816602 w_k1_IBM_s012
                                 + 0.0106999082054965 w_k1_IBM_s013
                                 + 0.0182044847189443 w_k1_IBM_s014
                                 + 0.0387823445959745 w_k1_IBM_s015
                                 + 0.0122897356096517 w_k1_IBM_s016
                                 + 0.0206544066802933 w_k1_IBM_s017
                                 + 0.0124017878029371 w_k1_IBM_s100
                                 + 0.0373665588428215 w_k1_IBM_s101
                                 + 0.00913447428703298 w_k1_IBM_s102
                                 + 0.00974707501082556 w_k1_IBM_s103
                                 + 0.0284231531879413 w_k1_IBM_s104
                                 + 0.0146542690018094 w_k1_IBM_s105
                                 + 0.00838326186439662 w_k1_IBM_s106
                                 + 0.0106246579724355 w_k1_IBM_s107
                                 + 0.0577482888877798 w_k1_IBM_s110
                                 + 0.0421805980179802 w_k1_IBM_s111
                                 + 0.0556981619386353 w_k1_IBM_s112
                                 + 0.0692773363094674 w_k1_IBM_s113
                                 + 0.0451626343129563 w_k1_IBM_s114
                                 + 0.0523746396232501 w_k1_IBM_s115
                                 + 0.0318924460042891 w_k1_IBM_s116
                                 + 0.0370381295075524 w_k1_IBM_s117
                                 + 0.0266951680494394 w_k1_INTC_s000
                                 + 0.0164853402315024 w_k1_INTC_s001
                                 + 0.053449264428158 w_k1_INTC_s002
                                 + 0.0278722126832061 w_k1_INTC_s003
                                 + 0.0688749315126862 w_k1_INTC_s004
                                 + 0.0296506295862322 w_k1_INTC_s005
                                 + 0.0557322469309213 w_k1_INTC_s006
                                 + 0.0268158979207917 w_k1_INTC_s007
                                 + 0.0277181157811304 w_k1_INTC_s010
                                 + 0.033580931493533 w_k1_INTC_s011
                                 + 0.0494066773274645 w_k1_INTC_s012
                                 + 0.00937657856016174 w_k1_INTC_s013
                                 + 0.0212860526077036 w_k1_INTC_s014
                                 + 0.0419038054739505 w_k1_INTC_s015
                                 + 0.0102987110203594 w_k1_INTC_s016
                                 + 0.0226684590054912 w_k1_INTC_s017
                                 + 0.0119087766510165 w_k1_INTC_s100
                                 + 0.0396595859755826 w_k1_INTC_s101
                                 + 0.00822676642893998 w_k1_INTC_s102
                                 + 0.00973331371807377 w_k1_INTC_s103
                                 + 0.0259666847415801 w_k1_INTC_s104
                                 + 0.0155511403948319 w_k1_INTC_s105
                                 + 0.00795641779431063 w_k1_INTC_s106
                                 + 0.0110213849978884 w_k1_INTC_s107
                                 + 0.0506253445986937 w_k1_INTC_s110
                                 + 0.0434277704686009 w_k1_INTC_s111
                                 + 0.0606035907807285 w_k1_INTC_s112
                                 + 0.0631843915337744 w_k1_INTC_s113
                                 + 0.0462332417167073 w_k1_INTC_s114
                                 + 0.0543910588660279 w_k1_INTC_s115
                                 + 0.0331373797971225 w_k1_INTC_s116
                                 + 0.0359055706789827 w_k1_INTC_s117
                                 + 0.029944730438566 w_k1_KO_s000
                                 + 0.0165437656205152 w_k1_KO_s001
                                 + 0.0488470351789367 w_k1_KO_s002
                                 + 0.0234724940996963 w_k1_KO_s003
                                 + 0.073692269728965 w_k1_KO_s004
                                 + 0.0313707174798137 w_k1_KO_s005
                                 + 0.0467347864094346 w_k1_KO_s006
                                 + 0.0262565991080315 w_k1_KO_s007
                                 + 0.02591765395621 w_k1_KO_s010
                                 + 0.0317231905043544 w_k1_KO_s011
                                 + 0.0504652558381092 w_k1_KO_s012
                                 + 0.0106028190778254 w_k1_KO_s013
                                 + 0.0172744287424182 w_k1_KO_s014
                                 + 0.0398410112208241 w_k1_KO_s015
                                 + 0.0125093181381224 w_k1_KO_s016
                                 + 0.0212649721023662 w_k1_KO_s017
                                 + 0.0124120815905532 w_k1_KO_s100
                                 + 0.035875920991293 w_k1_KO_s101
                                 + 0.00898331238334427 w_k1_KO_s102
                                 + 0.00957800232489581 w_k1_KO_s103
                                 + 0.0284341376325076 w_k1_KO_s104
                                 + 0.0133405769968866 w_k1_KO_s105
                                 + 0.00800615847221001 w_k1_KO_s106
                                 + 0.0103172135968538 w_k1_KO_s107
                                 + 0.0590262886174484 w_k1_KO_s110
                                 + 0.0395599719274205 w_k1_KO_s111
                                 + 0.0551946973877004 w_k1_KO_s112
                                 + 0.076605417009514 w_k1_KO_s113
                                 + 0.0430410279659979 w_k1_KO_s114
                                 + 0.0500077761213778 w_k1_KO_s115
                                 + 0.0291632622162646 w_k1_KO_s116
                                 + 0.0348677765816419 w_k1_KO_s117
                                 + 0.025919428431956 w_k1_MAT_s000
                                 + 0.0155760767458035 w_k1_MAT_s001
                                 + 0.0486779657429578 w_k1_MAT_s002
                                 + 0.0309910899583021 w_k1_MAT_s003
                                 + 0.0805935868690734 w_k1_MAT_s004
                                 + 0.0325857209919003 w_k1_MAT_s005
                                 + 0.0578848982168329 w_k1_MAT_s006
                                 + 0.0248032368309086 w_k1_MAT_s007
                                 + 0.0253649945137738 w_k1_MAT_s010
                                 + 0.0264501925007005 w_k1_MAT_s011
                                 + 0.0518965309076362 w_k1_MAT_s012
                                 + 0.0120090638228789 w_k1_MAT_s013
                                 + 0.0195522665264758 w_k1_MAT_s014
                                 + 0.0406146781128332 w_k1_MAT_s015
                                 + 0.0125375278315005 w_k1_MAT_s016
                                 + 0.0172761790841187 w_k1_MAT_s017
                                 + 0.014184814292732 w_k1_MAT_s100
                                 + 0.0492863339771538 w_k1_MAT_s101
                                 + 0.0125373788361248 w_k1_MAT_s102
                                 + 0.0104357969808919 w_k1_MAT_s103
                                 + 0.029872653600983 w_k1_MAT_s104
                                 + 0.0152676014696776 w_k1_MAT_s105
                                 + 0.00687106183675302 w_k1_MAT_s106
                                 + 0.0103471179348725 w_k1_MAT_s107
                                 + 0.0777311084264622 w_k1_MAT_s110
                                 + 0.0365831178680726 w_k1_MAT_s111
                                 + 0.0547988215859203 w_k1_MAT_s112
                                 + 0.0754050085243741 w_k1_MAT_s113
                                 + 0.0527555925300104 w_k1_MAT_s114
                                 + 0.0535507248416 w_k1_MAT_s115
                                 + 0.0242146362146933 w_k1_MAT_s116
                                 + 0.0344776506494315 w_k1_MAT_s117
                                 + 0.0266183973545715 w_k1_MSFT_s000
                                 + 0.0158303270872495 w_k1_MSFT_s001
                                 + 0.0534569781998865 w_k1_MSFT_s002
                                 + 0.0243942614651632 w_k1_MSFT_s003
                                 + 0.0755332000513779 w_k1_MSFT_s004
                                 + 0.0334955324776144 w_k1_MSFT_s005
                                 + 0.0534037273359164 w_k1_MSFT_s006
                                 + 0.0257835455452951 w_k1_MSFT_s007
                                 + 0.0268763533112852 w_k1_MSFT_s010
                                 + 0.0332278430667748 w_k1_MSFT_s011
                                 + 0.0489832369894049 w_k1_MSFT_s012
                                 + 0.00999189592735165 w_k1_MSFT_s013
                                 + 0.0167102406004154 w_k1_MSFT_s014
                                 + 0.0357558797184139 w_k1_MSFT_s015
                                 + 0.0111980306217435 w_k1_MSFT_s016
                                 + 0.0210917591542968 w_k1_MSFT_s017
                                 + 0.0121653180922991 w_k1_MSFT_s100
                                 + 0.0384533181191694 w_k1_MSFT_s101
                                 + 0.0092573645820459 w_k1_MSFT_s102
                                 + 0.0100812073892512 w_k1_MSFT_s103
                                 + 0.0274985724635027 w_k1_MSFT_s104
                                 + 0.013809146766206 w_k1_MSFT_s105
                                 + 0.00833680385897892 w_k1_MSFT_s106
                                 + 0.0107070088372799 w_k1_MSFT_s107
                                 + 0.058091417540872 w_k1_MSFT_s110
                                 + 0.0425523375930586 w_k1_MSFT_s111
                                 + 0.0526047208226132 w_k1_MSFT_s112
                                 + 0.0641384351521724 w_k1_MSFT_s113
                                 + 0.0463849061246824 w_k1_MSFT_s114
                                 + 0.0539815929803771 w_k1_MSFT_s115
                                 + 0.0301762241835665 w_k1_MSFT_s116
                                 + 0.0352275122415794 w_k1_MSFT_s117
                                 + 0.0277429882071071 w_k1_PG_s000
                                 + 0.0160949713963463 w_k1_PG_s001
                                 + 0.0451711721713977 w_k1_PG_s002
                                 + 0.0252750209943664 w_k1_PG_s003
                                 + 0.0753046920032309 w_k1_PG_s004
                                 + 0.029917932178042 w_k1_PG_s005
                                 + 0.0554579429211935 w_k1_PG_s006
                                 + 0.0256853787390101 w_k1_PG_s007
                                 + 0.02454416795888 w_k1_PG_s010
                                 + 0.0341379225951718 w_k1_PG_s011
                                 + 0.0505693865406875 w_k1_PG_s012
                                 + 0.0105803125346079 w_k1_PG_s013
                                 + 0.017126522476427 w_k1_PG_s014
                                 + 0.040447949257744 w_k1_PG_s015
                                 + 0.0119820241171411 w_k1_PG_s016
                                 + 0.0186837897051107 w_k1_PG_s017
                                 + 0.0119593098813535 w_k1_PG_s100
                                 + 0.0355589597773492 w_k1_PG_s101
                                 + 0.00897948309664521 w_k1_PG_s102
                                 + 0.00935097874027829 w_k1_PG_s103
                                 + 0.0262046947801913 w_k1_PG_s104
                                 + 0.0137029371869119 w_k1_PG_s105
                                 + 0.00787339123048943 w_k1_PG_s106
                                 + 0.0104699518255313 w_k1_PG_s107
                                 + 0.0572746082901866 w_k1_PG_s110
                                 + 0.0378703160342672 w_k1_PG_s111
                                 + 0.0551240242591624 w_k1_PG_s112
                                 + 0.0685248520396147 w_k1_PG_s113
                                 + 0.0422535002036245 w_k1_PG_s114
                                 + 0.0520335807348576 w_k1_PG_s115
                                 + 0.0292679103998311 w_k1_PG_s116
                                 + 0.038951578296821 w_k1_PG_s117 >= 1.219
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
