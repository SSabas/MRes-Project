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
 w_k1_ADM_s0_return_constraint:  1.05274 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.951464 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.22723 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.775994 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.881057 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.05781 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.35805 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.959733 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.915984 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.906583 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.799687 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.814533 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.10645 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.22632 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.832955 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.12137 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.02227 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.23716 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.12761 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.800338 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.12126 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.19923 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.06063 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.07618 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.834926 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.615736 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.725794 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.990253 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.32153 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.884428 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.993262 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.18224 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.01449 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.661004 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.27951 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.920065 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.786405 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.814017 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19992 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.898019 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.19729 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.765809 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.779289 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.0626 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.29698 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.986841 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.883892 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.902312 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.701561 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.724391 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.05296 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.50573 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.711461 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.13043 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.907354 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.63818 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.996322 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.726064 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.21621 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.989386 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.15027 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.19348 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.733424 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.602534 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.827324 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.25911 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.50332 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.726652 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.761699 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.22948 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.960019 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.648028 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.48506 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.991821 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.749426 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.940349 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.21014 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.852486 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.23697 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.73193 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.813151 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.17567 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.31515 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.871257 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.959383 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.936924 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.821101 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.648905 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.14372 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.62925 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.655945 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.0771 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.782969 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.6643 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.00487 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.682202 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.33289 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.979756 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.07993 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.31389 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.737434 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.463191 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.645588 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.12446 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.70963 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.719803 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.916826 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.24163 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.987904 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.500686 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.61911 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.695371 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.650681 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.898423 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10487 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.922429 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.14776 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.770301 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.881589 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0842 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.46048 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.898788 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.905465 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.916328 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.884399 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.822793 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.21306 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.3121 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.892803 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.00885 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.849774 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.44173 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.06606 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.743574 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.25017 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.993492 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.11172 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.12425 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.771583 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.655974 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.723207 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.02237 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.53827 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.812241 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.933064 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.15078 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.14414 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.574855 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.52679 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.861056 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.805756 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.88586 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18218 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.835852 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.09856 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.01516 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.73652 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.17128 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.690703 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.08495 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.90355 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.52524 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.73195 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.536081 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.835385 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.51851 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.520236 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.7849 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.16227 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.44068 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.664537 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.848135 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.04773 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.981098 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.853326 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.20661 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.00907 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.859578 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.43376 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.6486 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.08285 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.622427 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.731903 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.17278 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.641846 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.954384 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.948756 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.852801 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.661938 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.3319 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09352 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.951205 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.17115 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.880372 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.932736 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.07193 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.07268 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.86395 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.874714 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.932307 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.984317 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.729768 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.05151 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.104 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.838706 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.859072 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.888543 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.38587 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.964044 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.67054 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.12992 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.979509 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0067 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.05014 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.825667 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.679054 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.772735 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.00185 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.30656 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.770872 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.943535 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.03283 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.931054 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.732711 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.40556 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.922026 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.779846 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.01351 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21778 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.823477 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.05736 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.793096 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.808577 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23006 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.02537 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.92053 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.932412 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.21444 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.795509 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.687737 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04513 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.25102 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.675609 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.48168 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.990032 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.53983 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.888711 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.826452 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.24108 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.17023 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.967876 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.26762 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.960315 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.646124 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.0333 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.39102 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.30063 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.740909 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.852364 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.08347 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.752902 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.786185 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.2259 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.788938 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.711855 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.10053 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.22987 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.77461 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.967642 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.892752 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.752741 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.11429 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.12232 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.19538 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.1075 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.66613 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.827783 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.747584 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.19934 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.94075 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.490595 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.86722 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.08169 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.74239 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.719969 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.933219 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.18806 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.985177 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.00923 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.1759 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.82425 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.88438 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.58316 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.74965 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.2063 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.506104 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.745911 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.23573 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.603339 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.98878 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.17959 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.15653 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.854725 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.5507 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.08099 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.895914 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.12883 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.851638 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.930259 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.1491 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.25163 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.850204 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.989933 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.947842 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.868408 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.785383 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16788 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.27317 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.808479 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.00184 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.875891 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.38996 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.976475 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.700928 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.22972 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.05267 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03241 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.13773 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.802564 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.655737 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.817565 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.10223 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.51915 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.810313 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.981876 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.07941 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.03767 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.701591 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.40582 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.767606 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.810962 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.889352 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09832 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.901739 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.05789 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.916824 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.893807 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.04887 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.08679 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.950797 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.986412 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.05094 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.870829 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.862332 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.10313 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.23357 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.769038 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.12423 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.924833 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.18332 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.9871 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.807562 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.10411 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.921363 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.02852 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.08841 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.92651 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.809223 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.907441 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.14631 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.17957 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.798249 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.917598 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.1452 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02714 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.855596 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.26334 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.885788 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.845875 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05036 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.180999967337171 z_k1_s0000
                                 - 0.283000178477166 z_k1_s0001
                                 - 0.0819998846070224 z_k1_s0002
                                 - 0.168000347759991 z_k1_s0003
                                 - 0.165999936287172 z_k1_s0004
                                 - 0.0979998769937397 z_k1_s0005
                                 - 0.09199986457374 z_k1_s0006
                                 - 0.151999499974354 z_k1_s0007
                                 - 0.724000062700589 z_k1_s0010
                                 - 0.820001705118658 z_k1_s0011
                                 - 0.261999747472446 z_k1_s0012
                                 - 0.417999692178131 z_k1_s0013
                                 - 0.599002302352019 z_k1_s0014
                                 - 1.01600051801609 z_k1_s0015
                                 - 0.78899744516255 z_k1_s0016
                                 - 0.560998084792431 z_k1_s0017
                                 - 0.46399910919358 z_k1_s0020
                                 - 0.479999452027884 z_k1_s0021
                                 - 0.370001157417509 z_k1_s0022
                                 - 0.310999938353517 z_k1_s0023
                                 - 0.46699985053759 z_k1_s0024
                                 - 0.336999863534189 z_k1_s0025
                                 - 0.313999957497444 z_k1_s0026
                                 - 0.868001446509037 z_k1_s0027
                                 - 0.271000484502221 z_k1_s0030
                                 - 0.463000700461437 z_k1_s0031
                                 - 0.453999874845856 z_k1_s0032
                                 - 1.13100349746173 z_k1_s0033
                                 - 0.455999075424845 z_k1_s0034
                                 - 0.541000484112922 z_k1_s0035
                                 - 0.807000313361592 z_k1_s0036
                                 - 0.300000835024914 z_k1_s0037
                                 - 0.670999929952571 z_k1_s0040
                                 - 0.267999974832026 z_k1_s0041
                                 - 0.538001820409765 z_k1_s0042
                                 - 0.327000119810029 z_k1_s0043
                                 - 0.359000268730327 z_k1_s0044
                                 - 0.964999719469726 z_k1_s0045
                                 - 0.194999929638955 z_k1_s0046
                                 - 0.447001526511464 z_k1_s0047
                                 - 0.300999292726271 z_k1_s0050
                                 - 0.390998835164042 z_k1_s0051
                                 - 0.759995852341963 z_k1_s0052
                                 - 1.01899527345751 z_k1_s0053
                                 - 0.247999363520132 z_k1_s0054
                                 - 0.345999063945156 z_k1_s0055
                                 - 0.717996406430061 z_k1_s0056
                                 - 0.473998611849437 z_k1_s0057
                                 - 0.754000415427473 z_k1_s0060
                                 - 0.256000623602927 z_k1_s0061
                                 - 0.71500132702855 z_k1_s0062
                                 - 0.621000535392375 z_k1_s0063
                                 - 0.358001057453809 z_k1_s0064
                                 - 0.308000795805206 z_k1_s0065
                                 - 0.91000093203431 z_k1_s0066
                                 - 0.241000461848124 z_k1_s0067
                                 - 1.11599798511265 z_k1_s0070
                                 - 1.20300017923666 z_k1_s0071
                                 - 0.269999836004253 z_k1_s0072
                                 - 0.959999045504493 z_k1_s0073
                                 - 0.548999053732776 z_k1_s0074
                                 - 0.608001438183528 z_k1_s0075
                                 - 0.421999359901219 z_k1_s0076
                                 - 0.442999217156163 z_k1_s0077
                                 - 0.097000008081025 z_k1_s0100
                                 - 0.168000175090972 z_k1_s0101
                                 - 0.285999240782624 z_k1_s0102
                                 - 0.413000784823913 z_k1_s0103
                                 - 0.1050000352568 z_k1_s0104
                                 - 0.270001164432601 z_k1_s0105
                                 - 0.365000621769262 z_k1_s0106
                                 - 0.274000683520107 z_k1_s0107
                                 - 0.190999897973037 z_k1_s0110
                                 - 0.179999795675197 z_k1_s0111
                                 - 0.233998899138402 z_k1_s0112
                                 - 0.256998753361437 z_k1_s0113
                                 - 0.222000026412528 z_k1_s0114
                                 - 0.348000279024861 z_k1_s0115
                                 - 0.228000561774616 z_k1_s0116
                                 - 0.537999648171223 z_k1_s0117
                                 - 0.132000046557575 z_k1_s0120
                                 - 0.218999722142971 z_k1_s0121
                                 - 0.138000115673881 z_k1_s0122
                                 - 0.237000224292121 z_k1_s0123
                                 - 0.106000140120558 z_k1_s0124
                                 - 0.293000476310668 z_k1_s0125
                                 - 0.140000040445906 z_k1_s0126
                                 - 0.209000098282847 z_k1_s0127
                                 - 0.13300005777874 z_k1_s0130
                                 - 0.277000133230918 z_k1_s0131
                                 - 0.262000732767573 z_k1_s0132
                                 - 0.247999463000509 z_k1_s0133
                                 - 0.10900015953679 z_k1_s0134
                                 - 0.0740000721212816 z_k1_s0135
                                 - 0.357999639734894 z_k1_s0136
                                 - 0.254000252186564 z_k1_s0137
                                 - 0.332999650623498 z_k1_s0140
                                 - 0.634999828996398 z_k1_s0141
                                 - 0.208999969122418 z_k1_s0142
                                 - 0.697998987763575 z_k1_s0143
                                 - 0.317999663131451 z_k1_s0144
                                 - 0.221999886582245 z_k1_s0145
                                 - 0.958000921957479 z_k1_s0146
                                 - 0.21199989492088 z_k1_s0147
                                 - 0.499001316968812 z_k1_s0150
                                 - 0.251999337284835 z_k1_s0151
                                 - 0.412999600162302 z_k1_s0152
                                 - 0.213000407768416 z_k1_s0153
                                 - 0.233000394387907 z_k1_s0154
                                 - 0.246000262160389 z_k1_s0155
                                 - 0.145000183741737 z_k1_s0156
                                 - 0.245000790640217 z_k1_s0157
                                 - 0.478999567406606 z_k1_s0160
                                 - 0.924002906886545 z_k1_s0161
                                 - 0.192000644070042 z_k1_s0162
                                 - 0.379001094725272 z_k1_s0163
                                 - 0.600000085713557 z_k1_s0164
                                 - 0.300000813658908 z_k1_s0165
                                 - 0.217000454940908 z_k1_s0166
                                 - 0.763001612063227 z_k1_s0167
                                 - 0.178000430319087 z_k1_s0170
                                 - 0.319000496714701 z_k1_s0171
                                 - 0.18200043220265 z_k1_s0172
                                 - 0.436001591310463 z_k1_s0173
                                 - 0.815000383776068 z_k1_s0174
                                 - 0.583000967530403 z_k1_s0175
                                 - 0.225000452450958 z_k1_s0176
                                 - 0.726999649336651 z_k1_s0177
                                 - 0.761001609228009 z_k1_s1000
                                 - 0.636999782356938 z_k1_s1001
                                 - 0.852000801420806 z_k1_s1002
                                 - 0.529000699250046 z_k1_s1003
                                 - 0.467000873215982 z_k1_s1004
                                 - 1.05200111717704 z_k1_s1005
                                 - 0.71699990865943 z_k1_s1006
                                 - 0.422000394395278 z_k1_s1007
                                 - 0.0980001570003105 z_k1_s1010
                                 - 0.133000202693268 z_k1_s1011
                                 - 0.388000805442237 z_k1_s1012
                                 - 0.185000511363055 z_k1_s1013
                                 - 0.140000182771829 z_k1_s1014
                                 - 0.210000855358348 z_k1_s1015
                                 - 0.168000393686376 z_k1_s1016
                                 - 0.131000146113614 z_k1_s1017
                                 - 0.208000588571976 z_k1_s1020
                                 - 0.2129996295815 z_k1_s1021
                                 - 0.15599997717926 z_k1_s1022
                                 - 0.336998873403917 z_k1_s1023
                                 - 0.10399992288621 z_k1_s1024
                                 - 0.10399992288621 z_k1_s1025
                                 - 0.25800028366166 z_k1_s1026
                                 - 0.476999876653922 z_k1_s1027
                                 - 0.219000466588111 z_k1_s1030
                                 - 0.126000440530357 z_k1_s1031
                                 - 0.13500055356834 z_k1_s1032
                                 - 0.123999654076538 z_k1_s1033
                                 - 0.182000763210671 z_k1_s1034
                                 - 0.166000181587124 z_k1_s1035
                                 - 0.108000100254252 z_k1_s1036
                                 - 0.082000154367213 z_k1_s1037
                                 - 0.162000439385624 z_k1_s1040
                                 - 0.219000638613965 z_k1_s1041
                                 - 0.0740001233488744 z_k1_s1042
                                 - 0.090000136991855 z_k1_s1043
                                 - 0.147000577220552 z_k1_s1044
                                 - 0.0810000991926339 z_k1_s1045
                                 - 0.19799986758144 z_k1_s1046
                                 - 0.234000500779037 z_k1_s1047
                                 - 0.22199982108884 z_k1_s1050
                                 - 0.231001109675911 z_k1_s1051
                                 - 0.0990001836319952 z_k1_s1052
                                 - 0.244000001629541 z_k1_s1053
                                 - 0.1030001450029 z_k1_s1054
                                 - 0.156000222668476 z_k1_s1055
                                 - 0.30100082666747 z_k1_s1056
                                 - 0.216001058034655 z_k1_s1057
                                 - 0.0890000954296755 z_k1_s1060
                                 - 0.0870000932851885 z_k1_s1061
                                 - 0.102000109368842 z_k1_s1062
                                 - 0.121000588142144 z_k1_s1063
                                 - 0.143999696002396 z_k1_s1064
                                 - 0.243000489755519 z_k1_s1065
                                 - 0.176000647115539 z_k1_s1066
                                 - 0.183999738892137 z_k1_s1067
                                 - 0.23200003852432 z_k1_s1070
                                 - 0.0800001353843906 z_k1_s1071
                                 - 0.123000020424532 z_k1_s1072
                                 - 0.0930001375430809 z_k1_s1073
                                 - 0.118000141694408 z_k1_s1074
                                 - 0.255000042343541 z_k1_s1075
                                 - 0.194000032214302 z_k1_s1076
                                 - 0.126000020922691 z_k1_s1077
                                 - 0.206999857847439 z_k1_s1100
                                 - 0.162999919661493 z_k1_s1101
                                 - 0.246999164562274 z_k1_s1102
                                 - 0.338000332137064 z_k1_s1103
                                 - 0.321998740652124 z_k1_s1104
                                 - 0.339000139605028 z_k1_s1105
                                 - 0.180999958082982 z_k1_s1106
                                 - 0.538999009177961 z_k1_s1107
                                 - 1.35299389332403 z_k1_s1110
                                 - 1.9439985609894 z_k1_s1111
                                 - 0.596998745730502 z_k1_s1112
                                 - 1.04499953698955 z_k1_s1113
                                 - 0.657998772619734 z_k1_s1114
                                 - 0.790998567345976 z_k1_s1115
                                 - 0.661998312556036 z_k1_s1116
                                 - 0.998999595882758 z_k1_s1117
                                 - 0.161999991580665 z_k1_s1120
                                 - 0.205000369660784 z_k1_s1121
                                 - 0.237000755227403 z_k1_s1122
                                 - 0.229999504009502 z_k1_s1123
                                 - 0.13300006223599 z_k1_s1124
                                 - 0.504000471673267 z_k1_s1125
                                 - 0.281000351881341 z_k1_s1126
                                 - 0.114999938704784 z_k1_s1127
                                 - 0.208000076658668 z_k1_s1130
                                 - 0.230999978135372 z_k1_s1131
                                 - 0.753999721487799 z_k1_s1132
                                 - 0.313998703325431 z_k1_s1133
                                 - 0.398001173883174 z_k1_s1134
                                 - 0.467000321913416 z_k1_s1135
                                 - 0.231999828703958 z_k1_s1136
                                 - 0.177999830202184 z_k1_s1137
                                 - 0.355999211494536 z_k1_s1140
                                 - 0.571000452748083 z_k1_s1141
                                 - 0.284999553489596 z_k1_s1142
                                 - 0.403998890645962 z_k1_s1143
                                 - 0.789998166797049 z_k1_s1144
                                 - 0.242999462970425 z_k1_s1145
                                 - 0.54600075245017 z_k1_s1146
                                 - 1.04799744994622 z_k1_s1147
                                 - 0.200999833980227 z_k1_s1150
                                 - 0.161999970660118 z_k1_s1151
                                 - 0.142000013560079 z_k1_s1152
                                 - 0.5250001995002 z_k1_s1153
                                 - 0.237999383440529 z_k1_s1154
                                 - 0.47699987826037 z_k1_s1155
                                 - 0.216000385079895 z_k1_s1156
                                 - 0.15999986890018 z_k1_s1157
                                 - 0.426998742311656 z_k1_s1160
                                 - 0.176999857489695 z_k1_s1161
                                 - 0.696998626278947 z_k1_s1162
                                 - 0.30300050100749 z_k1_s1163
                                 - 0.439999257206173 z_k1_s1164
                                 - 0.310000088258655 z_k1_s1165
                                 - 0.310000088258655 z_k1_s1166
                                 - 0.325999571975435 z_k1_s1167
                                 - 1.10700228812463 z_k1_s1170
                                 - 1.25599764847793 z_k1_s1171
                                 - 0.455999196313582 z_k1_s1172
                                 - 1.04699867166675 z_k1_s1173
                                 - 1.37299527098203 z_k1_s1174
                                 - 0.993996524861047 z_k1_s1175
                                 - 0.571999639806616 z_k1_s1176
                                 - 1.05300217650879 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.21395 w_k1_ADM_s000
                                 + 1.35731 w_k1_AXP_s000
                                 + 1.47211 w_k1_DIS_s000
                                 + 1.14841 w_k1_FDX_s000
                                 + 1.33763 w_k1_IBM_s000
                                 + 1.14532 w_k1_INTC_s000
                                 + 1.35143 w_k1_KO_s000 + 1.00068 w_k1_MAT_s000
                                 + 1.1913 w_k1_MSFT_s000 + 1.2122 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.00018 w_k1_ADM_s000
                                 + 0.891159 w_k1_AXP_s000
                                 + 0.905705 w_k1_DIS_s000
                                 + 0.965209 w_k1_FDX_s000
                                 + 0.79698 w_k1_IBM_s000
                                 + 1.13562 w_k1_INTC_s000
                                 + 0.78448 w_k1_KO_s000
                                 + 0.526503 w_k1_MAT_s000
                                 + 1.02721 w_k1_MSFT_s000
                                 + 0.94895 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.14567 w_k1_ADM_s000
                                 + 1.59033 w_k1_AXP_s000
                                 + 1.96147 w_k1_DIS_s000 + 1.526 w_k1_FDX_s000
                                 + 1.34518 w_k1_IBM_s000
                                 + 1.58218 w_k1_INTC_s000
                                 + 1.35983 w_k1_KO_s000
                                 + 0.941363 w_k1_MAT_s000
                                 + 1.46133 w_k1_MSFT_s000
                                 + 1.33156 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.14316 w_k1_ADM_s000
                                 + 1.02752 w_k1_AXP_s000
                                 + 1.27493 w_k1_DIS_s000
                                 + 1.20295 w_k1_FDX_s000
                                 + 0.817393 w_k1_IBM_s000
                                 + 1.33143 w_k1_INTC_s000
                                 + 0.929638 w_k1_KO_s000
                                 + 0.605678 w_k1_MAT_s000
                                 + 1.31511 w_k1_MSFT_s000
                                 + 1.10734 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.617654 w_k1_ADM_s000
                                 + 0.665264 w_k1_AXP_s000
                                 + 0.653737 w_k1_DIS_s000
                                 + 0.747913 w_k1_FDX_s000
                                 + 0.669031 w_k1_IBM_s000
                                 + 0.992683 w_k1_INTC_s000
                                 + 0.810819 w_k1_KO_s000
                                 + 0.535643 w_k1_MAT_s000
                                 + 0.938336 w_k1_MSFT_s000
                                 + 0.830801 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.770612 w_k1_ADM_s000
                                 + 1.2047 w_k1_AXP_s000
                                 + 0.763623 w_k1_DIS_s000
                                 + 0.842492 w_k1_FDX_s000
                                 + 0.876165 w_k1_IBM_s000
                                 + 1.02294 w_k1_INTC_s000
                                 + 0.95382 w_k1_KO_s000 + 1.31643 w_k1_MAT_s000
                                 + 0.92656 w_k1_MSFT_s000
                                 + 1.14507 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.05704 w_k1_ADM_s000
                                 + 1.09163 w_k1_AXP_s000
                                 + 0.989902 w_k1_DIS_s000
                                 + 1.0262 w_k1_FDX_s000
                                 + 0.846642 w_k1_IBM_s000
                                 + 0.971496 w_k1_INTC_s000
                                 + 0.930439 w_k1_KO_s000
                                 + 0.863727 w_k1_MAT_s000
                                 + 0.99939 w_k1_MSFT_s000
                                 + 0.99882 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.05853 w_k1_ADM_s000
                                 + 1.10459 w_k1_AXP_s000
                                 + 1.03301 w_k1_DIS_s000
                                 + 0.976999 w_k1_FDX_s000
                                 + 1.4624 w_k1_IBM_s000
                                 + 1.11995 w_k1_INTC_s000
                                 + 1.14216 w_k1_KO_s000 + 1.33314 w_k1_MAT_s000
                                 + 1.1857 w_k1_MSFT_s000 + 1.19838 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.24162 w_k1_ADM_s001
                                 + 1.36941 w_k1_AXP_s001
                                 + 1.58753 w_k1_DIS_s001
                                 + 1.36015 w_k1_FDX_s001
                                 + 1.28375 w_k1_IBM_s001
                                 + 1.60782 w_k1_INTC_s001
                                 + 1.47163 w_k1_KO_s001 + 1.4926 w_k1_MAT_s001
                                 + 1.57534 w_k1_MSFT_s001
                                 + 1.27448 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.8189 w_k1_ADM_s001
                                 + 0.895049 w_k1_AXP_s001
                                 + 0.73406 w_k1_DIS_s001
                                 + 0.977804 w_k1_FDX_s001
                                 + 0.743225 w_k1_IBM_s001
                                 + 0.798505 w_k1_INTC_s001
                                 + 0.812763 w_k1_KO_s001
                                 + 0.918679 w_k1_MAT_s001
                                 + 1.05767 w_k1_MSFT_s001
                                 + 0.985924 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.600995 w_k1_ADM_s001
                                 + 0.638828 w_k1_AXP_s001
                                 + 0.465969 w_k1_DIS_s001
                                 + 0.615812 w_k1_FDX_s001
                                 + 0.722142 w_k1_IBM_s001
                                 + 0.787396 w_k1_INTC_s001
                                 + 0.706331 w_k1_KO_s001
                                 + 0.739834 w_k1_MAT_s001
                                 + 0.697222 w_k1_MSFT_s001
                                 + 0.795155 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.900295 w_k1_ADM_s001
                                 + 0.882051 w_k1_AXP_s001
                                 + 0.905979 w_k1_DIS_s001
                                 + 0.842327 w_k1_FDX_s001
                                 + 1.25212 w_k1_IBM_s001
                                 + 0.859668 w_k1_INTC_s001
                                 + 1.09083 w_k1_KO_s001 + 1.09314 w_k1_MAT_s001
                                 + 0.922955 w_k1_MSFT_s001
                                 + 1.03457 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.916424 w_k1_ADM_s001
                                 + 0.791313 w_k1_AXP_s001
                                 + 1.13677 w_k1_DIS_s001
                                 + 0.989442 w_k1_FDX_s001
                                 + 0.994289 w_k1_IBM_s001
                                 + 0.943623 w_k1_INTC_s001
                                 + 1.07596 w_k1_KO_s001 + 1.06711 w_k1_MAT_s001
                                 + 1.07047 w_k1_MSFT_s001 + 1.0192 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.945403 w_k1_ADM_s001
                                 + 1.24299 w_k1_AXP_s001
                                 + 0.992339 w_k1_DIS_s001
                                 + 1.1589 w_k1_FDX_s001 + 0.96552 w_k1_IBM_s001
                                 + 0.902375 w_k1_INTC_s001
                                 + 1.0223 w_k1_KO_s001 + 1.36137 w_k1_MAT_s001
                                 + 1.04774 w_k1_MSFT_s001
                                 + 1.02339 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.730632 w_k1_ADM_s001
                                 + 0.733048 w_k1_AXP_s001
                                 + 0.662436 w_k1_DIS_s001
                                 + 0.750305 w_k1_FDX_s001
                                 + 0.938374 w_k1_IBM_s001
                                 + 0.797507 w_k1_INTC_s001
                                 + 0.826725 w_k1_KO_s001
                                 + 0.984656 w_k1_MAT_s001
                                 + 0.909625 w_k1_MSFT_s001
                                 + 0.883673 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.09269 w_k1_ADM_s001
                                 + 1.3327 w_k1_AXP_s001 + 1.22284 w_k1_DIS_s001
                                 + 1.10224 w_k1_FDX_s001
                                 + 1.38005 w_k1_IBM_s001
                                 + 1.11458 w_k1_INTC_s001
                                 + 1.35914 w_k1_KO_s001 + 2.21502 w_k1_MAT_s001
                                 + 1.1935 w_k1_MSFT_s001 + 1.19193 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.86982 w_k1_ADM_s002
                                 + 0.701496 w_k1_AXP_s002
                                 + 0.644522 w_k1_DIS_s002
                                 + 0.808782 w_k1_FDX_s002
                                 + 0.675503 w_k1_IBM_s002
                                 + 0.865957 w_k1_INTC_s002
                                 + 0.791314 w_k1_KO_s002
                                 + 0.508766 w_k1_MAT_s002
                                 + 0.837584 w_k1_MSFT_s002
                                 + 0.828305 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.669262 w_k1_ADM_s002
                                 + 0.810719 w_k1_AXP_s002
                                 + 1.15957 w_k1_DIS_s002
                                 + 1.01944 w_k1_FDX_s002
                                 + 1.24235 w_k1_IBM_s002
                                 + 0.90272 w_k1_INTC_s002
                                 + 1.18584 w_k1_KO_s002 + 1.08017 w_k1_MAT_s002
                                 + 1.00944 w_k1_MSFT_s002
                                 + 1.07219 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.26854 w_k1_ADM_s002
                                 + 1.21062 w_k1_AXP_s002
                                 + 1.32558 w_k1_DIS_s002
                                 + 1.34741 w_k1_FDX_s002
                                 + 0.916146 w_k1_IBM_s002
                                 + 1.24181 w_k1_INTC_s002
                                 + 1.25011 w_k1_KO_s002 + 1.07387 w_k1_MAT_s002
                                 + 1.31386 w_k1_MSFT_s002
                                 + 1.19193 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.36572 w_k1_ADM_s002
                                 + 1.2926 w_k1_AXP_s002 + 1.00867 w_k1_DIS_s002
                                 + 1.20378 w_k1_FDX_s002
                                 + 0.994438 w_k1_IBM_s002
                                 + 1.24988 w_k1_INTC_s002
                                 + 1.02035 w_k1_KO_s002
                                 + 0.876006 w_k1_MAT_s002
                                 + 1.05563 w_k1_MSFT_s002
                                 + 1.08104 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.36667 w_k1_ADM_s002
                                 + 1.43546 w_k1_AXP_s002 + 1.6393 w_k1_DIS_s002
                                 + 1.43168 w_k1_FDX_s002
                                 + 1.35883 w_k1_IBM_s002
                                 + 1.29978 w_k1_INTC_s002
                                 + 1.55176 w_k1_KO_s002 + 1.34152 w_k1_MAT_s002
                                 + 1.34528 w_k1_MSFT_s002
                                 + 1.31024 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.926136 w_k1_ADM_s002
                                 + 0.873857 w_k1_AXP_s002
                                 + 0.954033 w_k1_DIS_s002
                                 + 1.03485 w_k1_FDX_s002
                                 + 0.76068 w_k1_IBM_s002
                                 + 0.866956 w_k1_INTC_s002
                                 + 0.919198 w_k1_KO_s002
                                 + 0.829787 w_k1_MAT_s002
                                 + 1.02398 w_k1_MSFT_s002
                                 + 0.977863 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.769634 w_k1_ADM_s002
                                 + 1.00296 w_k1_AXP_s002
                                 + 0.789309 w_k1_DIS_s002
                                 + 0.918459 w_k1_FDX_s002
                                 + 1.28883 w_k1_IBM_s002
                                 + 0.911789 w_k1_INTC_s002
                                 + 1.04122 w_k1_KO_s002 + 1.34391 w_k1_MAT_s002
                                 + 0.825795 w_k1_MSFT_s002
                                 + 1.00511 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.999657 w_k1_ADM_s002
                                 + 0.752457 w_k1_AXP_s002
                                 + 0.828333 w_k1_DIS_s002
                                 + 0.943686 w_k1_FDX_s002
                                 + 0.941704 w_k1_IBM_s002
                                 + 0.94657 w_k1_INTC_s002
                                 + 0.942375 w_k1_KO_s002
                                 + 0.561586 w_k1_MAT_s002
                                 + 0.873851 w_k1_MSFT_s002
                                 + 0.987999 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.21757 w_k1_ADM_s003
                                 + 1.67626 w_k1_AXP_s003
                                 + 1.85855 w_k1_DIS_s003
                                 + 1.40377 w_k1_FDX_s003
                                 + 1.19167 w_k1_IBM_s003
                                 + 1.24578 w_k1_INTC_s003 + 1.25 w_k1_KO_s003
                                 + 1.21602 w_k1_MAT_s003
                                 + 1.34087 w_k1_MSFT_s003
                                 + 1.24085 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.755192 w_k1_ADM_s003
                                 + 0.73377 w_k1_AXP_s003
                                 + 0.610066 w_k1_DIS_s003
                                 + 0.729682 w_k1_FDX_s003
                                 + 0.648897 w_k1_IBM_s003
                                 + 0.868403 w_k1_INTC_s003
                                 + 0.686796 w_k1_KO_s003
                                 + 0.501658 w_k1_MAT_s003
                                 + 0.82202 w_k1_MSFT_s003
                                 + 0.786672 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.983808 w_k1_ADM_s003
                                 + 0.945192 w_k1_AXP_s003
                                 + 0.774124 w_k1_DIS_s003
                                 + 0.786924 w_k1_FDX_s003
                                 + 1.05661 w_k1_IBM_s003
                                 + 0.729655 w_k1_INTC_s003
                                 + 0.897931 w_k1_KO_s003
                                 + 1.12983 w_k1_MAT_s003
                                 + 0.744934 w_k1_MSFT_s003
                                 + 0.866044 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.06347 w_k1_ADM_s003
                                 + 1.30999 w_k1_AXP_s003
                                 + 1.13947 w_k1_DIS_s003
                                 + 1.10172 w_k1_FDX_s003
                                 + 0.987409 w_k1_IBM_s003
                                 + 1.00921 w_k1_INTC_s003
                                 + 0.927558 w_k1_KO_s003
                                 + 1.07048 w_k1_MAT_s003
                                 + 1.00159 w_k1_MSFT_s003
                                 + 1.01019 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.76017 w_k1_ADM_s003
                                 + 0.84316 w_k1_AXP_s003
                                 + 0.78551 w_k1_DIS_s003
                                 + 0.772649 w_k1_FDX_s003
                                 + 0.861863 w_k1_IBM_s003
                                 + 0.796152 w_k1_INTC_s003
                                 + 0.861443 w_k1_KO_s003
                                 + 0.921135 w_k1_MAT_s003
                                 + 0.902869 w_k1_MSFT_s003
                                 + 0.80881 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.897784 w_k1_ADM_s003
                                 + 0.949834 w_k1_AXP_s003
                                 + 1.16593 w_k1_DIS_s003
                                 + 0.957972 w_k1_FDX_s003
                                 + 0.758491 w_k1_IBM_s003
                                 + 0.968377 w_k1_INTC_s003
                                 + 0.921162 w_k1_KO_s003
                                 + 0.917758 w_k1_MAT_s003
                                 + 1.115 w_k1_MSFT_s003 + 0.8856 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.896285 w_k1_ADM_s003
                                 + 1.19656 w_k1_AXP_s003
                                 + 1.22925 w_k1_DIS_s003
                                 + 1.02335 w_k1_FDX_s003
                                 + 1.43255 w_k1_IBM_s003
                                 + 1.18401 w_k1_INTC_s003
                                 + 1.22523 w_k1_KO_s003 + 1.27594 w_k1_MAT_s003
                                 + 1.05456 w_k1_MSFT_s003
                                 + 1.06367 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.15599 w_k1_ADM_s003
                                 + 0.905903 w_k1_AXP_s003
                                 + 0.861289 w_k1_DIS_s003
                                 + 0.973785 w_k1_FDX_s003
                                 + 0.615456 w_k1_IBM_s003
                                 + 0.889254 w_k1_INTC_s003
                                 + 0.778716 w_k1_KO_s003
                                 + 0.676072 w_k1_MAT_s003
                                 + 1.07959 w_k1_MSFT_s003
                                 + 0.948783 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.29325 w_k1_ADM_s004
                                 + 1.20108 w_k1_AXP_s004 + 1.486 w_k1_DIS_s004
                                 + 1.23209 w_k1_FDX_s004
                                 + 1.30058 w_k1_IBM_s004
                                 + 0.971311 w_k1_INTC_s004
                                 + 1.28443 w_k1_KO_s004 + 1.21082 w_k1_MAT_s004
                                 + 1.22678 w_k1_MSFT_s004
                                 + 1.09397 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.03929 w_k1_ADM_s004
                                 + 1.10776 w_k1_AXP_s004
                                 + 0.887151 w_k1_DIS_s004
                                 + 0.904248 w_k1_FDX_s004
                                 + 1.37921 w_k1_IBM_s004
                                 + 0.844325 w_k1_INTC_s004
                                 + 1.1668 w_k1_KO_s004 + 1.29557 w_k1_MAT_s004
                                 + 0.957193 w_k1_MSFT_s004
                                 + 1.13435 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.832798 w_k1_ADM_s004
                                 + 0.637277 w_k1_AXP_s004
                                 + 0.743393 w_k1_DIS_s004
                                 + 0.794674 w_k1_FDX_s004
                                 + 0.957859 w_k1_IBM_s004
                                 + 0.716068 w_k1_INTC_s004
                                 + 0.90766 w_k1_KO_s004
                                 + 0.924385 w_k1_MAT_s004
                                 + 0.86458 w_k1_MSFT_s004
                                 + 0.89467 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.858795 w_k1_ADM_s004
                                 + 0.816002 w_k1_AXP_s004
                                 + 0.571154 w_k1_DIS_s004
                                 + 0.73427 w_k1_FDX_s004
                                 + 0.743789 w_k1_IBM_s004
                                 + 0.718657 w_k1_INTC_s004
                                 + 0.634986 w_k1_KO_s004
                                 + 0.675123 w_k1_MAT_s004
                                 + 0.701984 w_k1_MSFT_s004
                                 + 0.765469 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.794335 w_k1_ADM_s004
                                 + 0.956447 w_k1_AXP_s004
                                 + 1.07669 w_k1_DIS_s004
                                 + 0.939641 w_k1_FDX_s004
                                 + 1.16481 w_k1_IBM_s004
                                 + 0.972244 w_k1_INTC_s004
                                 + 1.22849 w_k1_KO_s004 + 1.03793 w_k1_MAT_s004
                                 + 1.10616 w_k1_MSFT_s004
                                 + 0.989997 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.03522 w_k1_ADM_s004
                                 + 1.08785 w_k1_AXP_s004
                                 + 1.16089 w_k1_DIS_s004
                                 + 1.22794 w_k1_FDX_s004
                                 + 0.803872 w_k1_IBM_s004
                                 + 0.959231 w_k1_INTC_s004
                                 + 1.1054 w_k1_KO_s004 + 0.956212 w_k1_MAT_s004
                                 + 1.3273 w_k1_MSFT_s004 + 1.0673 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.833231 w_k1_ADM_s004
                                 + 0.648444 w_k1_AXP_s004
                                 + 0.709167 w_k1_DIS_s004
                                 + 0.749885 w_k1_FDX_s004
                                 + 0.771552 w_k1_IBM_s004
                                 + 0.723276 w_k1_INTC_s004
                                 + 0.823514 w_k1_KO_s004
                                 + 0.56142 w_k1_MAT_s004
                                 + 0.978625 w_k1_MSFT_s004
                                 + 0.867917 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.931713 w_k1_ADM_s004
                                 + 1.00983 w_k1_AXP_s004
                                 + 0.678519 w_k1_DIS_s004
                                 + 0.838666 w_k1_FDX_s004
                                 + 1.06715 w_k1_IBM_s004
                                 + 0.81978 w_k1_INTC_s004
                                 + 0.940825 w_k1_KO_s004
                                 + 0.922581 w_k1_MAT_s004
                                 + 0.929576 w_k1_MSFT_s004
                                 + 0.943786 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.977328 w_k1_ADM_s005
                                 + 1.13882 w_k1_AXP_s005
                                 + 0.964419 w_k1_DIS_s005
                                 + 1.16898 w_k1_FDX_s005
                                 + 0.81477 w_k1_IBM_s005
                                 + 1.11393 w_k1_INTC_s005
                                 + 0.816508 w_k1_KO_s005
                                 + 0.806818 w_k1_MAT_s005
                                 + 1.03357 w_k1_MSFT_s005
                                 + 0.998663 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.33811 w_k1_ADM_s005
                                 + 1.48634 w_k1_AXP_s005
                                 + 1.74956 w_k1_DIS_s005
                                 + 1.47968 w_k1_FDX_s005
                                 + 1.08597 w_k1_IBM_s005
                                 + 1.43811 w_k1_INTC_s005
                                 + 1.33242 w_k1_KO_s005 + 1.19973 w_k1_MAT_s005
                                 + 1.52301 w_k1_MSFT_s005
                                 + 1.22887 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.22231 w_k1_ADM_s005
                                 + 1.41532 w_k1_AXP_s005
                                 + 1.29012 w_k1_DIS_s005
                                 + 1.17807 w_k1_FDX_s005
                                 + 1.38399 w_k1_IBM_s005
                                 + 1.25163 w_k1_INTC_s005
                                 + 1.11803 w_k1_KO_s005 + 1.20059 w_k1_MAT_s005
                                 + 1.19577 w_k1_MSFT_s005
                                 + 1.11694 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.902831 w_k1_ADM_s005
                                 + 0.834824 w_k1_AXP_s005
                                 + 0.908513 w_k1_DIS_s005
                                 + 0.893646 w_k1_FDX_s005
                                 + 1.26051 w_k1_IBM_s005
                                 + 0.895974 w_k1_INTC_s005
                                 + 0.957643 w_k1_KO_s005
                                 + 0.728665 w_k1_MAT_s005
                                 + 0.961152 w_k1_MSFT_s005
                                 + 0.913937 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.795724 w_k1_ADM_s005
                                 + 0.633252 w_k1_AXP_s005
                                 + 0.576325 w_k1_DIS_s005
                                 + 0.754195 w_k1_FDX_s005
                                 + 0.553322 w_k1_IBM_s005
                                 + 0.718294 w_k1_INTC_s005
                                 + 0.569988 w_k1_KO_s005
                                 + 0.502884 w_k1_MAT_s005
                                 + 0.75362 w_k1_MSFT_s005
                                 + 0.711344 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.28553 w_k1_ADM_s005
                                 + 0.93645 w_k1_AXP_s005
                                 + 1.33829 w_k1_DIS_s005
                                 + 1.25162 w_k1_FDX_s005
                                 + 0.858785 w_k1_IBM_s005
                                 + 0.919494 w_k1_INTC_s005
                                 + 0.953528 w_k1_KO_s005
                                 + 0.708192 w_k1_MAT_s005
                                 + 1.21206 w_k1_MSFT_s005 + 1.0987 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.906893 w_k1_ADM_s005
                                 + 0.805064 w_k1_AXP_s005
                                 + 0.737288 w_k1_DIS_s005
                                 + 0.911494 w_k1_FDX_s005
                                 + 0.830799 w_k1_IBM_s005
                                 + 0.937306 w_k1_INTC_s005
                                 + 0.870058 w_k1_KO_s005
                                 + 0.877503 w_k1_MAT_s005
                                 + 1.0043 w_k1_MSFT_s005
                                 + 0.857622 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.737336 w_k1_ADM_s005
                                 + 0.675197 w_k1_AXP_s005
                                 + 0.609156 w_k1_DIS_s005
                                 + 0.72309 w_k1_FDX_s005
                                 + 1.07131 w_k1_IBM_s005
                                 + 0.941687 w_k1_INTC_s005
                                 + 0.769036 w_k1_KO_s005
                                 + 0.681563 w_k1_MAT_s005
                                 + 0.752386 w_k1_MSFT_s005
                                 + 0.792368 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.04322 w_k1_ADM_s006
                                 + 1.32629 w_k1_AXP_s006
                                 + 1.29352 w_k1_DIS_s006
                                 + 1.10903 w_k1_FDX_s006
                                 + 1.25728 w_k1_IBM_s006
                                 + 1.01431 w_k1_INTC_s006 + 1.1024 w_k1_KO_s006
                                 + 1.01638 w_k1_MAT_s006
                                 + 1.06625 w_k1_MSFT_s006
                                 + 1.06022 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.709967 w_k1_ADM_s006
                                 + 1.09565 w_k1_AXP_s006
                                 + 0.932187 w_k1_DIS_s006
                                 + 0.930771 w_k1_FDX_s006
                                 + 1.37861 w_k1_IBM_s006
                                 + 0.994766 w_k1_INTC_s006
                                 + 1.01184 w_k1_KO_s006 + 1.35931 w_k1_MAT_s006
                                 + 0.876137 w_k1_MSFT_s006
                                 + 0.902218 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.834441 w_k1_ADM_s006
                                 + 0.729163 w_k1_AXP_s006
                                 + 0.628977 w_k1_DIS_s006
                                 + 0.767737 w_k1_FDX_s006
                                 + 0.752322 w_k1_IBM_s006
                                 + 0.853387 w_k1_INTC_s006
                                 + 0.72672 w_k1_KO_s006
                                 + 0.564682 w_k1_MAT_s006
                                 + 0.80082 w_k1_MSFT_s006
                                 + 0.802097 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.96991 w_k1_ADM_s006
                                 + 0.893622 w_k1_AXP_s006
                                 + 0.917109 w_k1_DIS_s006
                                 + 0.95514 w_k1_FDX_s006
                                 + 0.981538 w_k1_IBM_s006
                                 + 1.02931 w_k1_INTC_s006
                                 + 0.920464 w_k1_KO_s006
                                 + 0.859005 w_k1_MAT_s006
                                 + 1.00909 w_k1_MSFT_s006
                                 + 0.896302 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.923005 w_k1_ADM_s006
                                 + 1.2486 w_k1_AXP_s006 + 1.16023 w_k1_DIS_s006
                                 + 0.970101 w_k1_FDX_s006
                                 + 1.14233 w_k1_IBM_s006
                                 + 1.08462 w_k1_INTC_s006
                                 + 1.19686 w_k1_KO_s006 + 1.49025 w_k1_MAT_s006
                                 + 1.08769 w_k1_MSFT_s006
                                 + 1.15584 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.858453 w_k1_ADM_s006
                                 + 0.773582 w_k1_AXP_s006
                                 + 0.910289 w_k1_DIS_s006
                                 + 0.766718 w_k1_FDX_s006
                                 + 0.998141 w_k1_IBM_s006
                                 + 0.914824 w_k1_INTC_s006
                                 + 0.939912 w_k1_KO_s006
                                 + 0.630606 w_k1_MAT_s006
                                 + 0.970673 w_k1_MSFT_s006
                                 + 0.858429 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.995103 w_k1_ADM_s006
                                 + 0.998496 w_k1_AXP_s006
                                 + 0.927039 w_k1_DIS_s006
                                 + 0.999234 w_k1_FDX_s006
                                 + 0.71773 w_k1_IBM_s006
                                 + 1.0357 w_k1_INTC_s006
                                 + 0.774385 w_k1_KO_s006
                                 + 0.558923 w_k1_MAT_s006
                                 + 0.930279 w_k1_MSFT_s006
                                 + 0.861908 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.0409 w_k1_ADM_s006
                                 + 1.60189 w_k1_AXP_s006
                                 + 1.81223 w_k1_DIS_s006
                                 + 1.46644 w_k1_FDX_s006
                                 + 1.26022 w_k1_IBM_s006
                                 + 1.26725 w_k1_INTC_s006
                                 + 1.44541 w_k1_KO_s006 + 1.53997 w_k1_MAT_s006
                                 + 1.328 w_k1_MSFT_s006 + 1.21361 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.01066 w_k1_ADM_s007
                                 + 0.89212 w_k1_AXP_s007
                                 + 1.09139 w_k1_DIS_s007
                                 + 0.959241 w_k1_FDX_s007
                                 + 1.11405 w_k1_IBM_s007
                                 + 0.882806 w_k1_INTC_s007
                                 + 1.17305 w_k1_KO_s007 + 1.02263 w_k1_MAT_s007
                                 + 1.05058 w_k1_MSFT_s007
                                 + 1.03363 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.15364 w_k1_ADM_s007
                                 + 1.03661 w_k1_AXP_s007
                                 + 1.07973 w_k1_DIS_s007
                                 + 1.13949 w_k1_FDX_s007
                                 + 0.804424 w_k1_IBM_s007
                                 + 1.03784 w_k1_INTC_s007
                                 + 1.14284 w_k1_KO_s007
                                 + 0.947395 w_k1_MAT_s007
                                 + 1.10489 w_k1_MSFT_s007
                                 + 1.05143 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.960264 w_k1_ADM_s007
                                 + 0.805058 w_k1_AXP_s007
                                 + 0.921797 w_k1_DIS_s007
                                 + 1.02972 w_k1_FDX_s007
                                 + 0.52963 w_k1_IBM_s007
                                 + 1.06083 w_k1_INTC_s007
                                 + 0.902627 w_k1_KO_s007
                                 + 0.51508 w_k1_MAT_s007
                                 + 1.00613 w_k1_MSFT_s007
                                 + 0.817352 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.851701 w_k1_ADM_s007
                                 + 1.30965 w_k1_AXP_s007
                                 + 1.49138 w_k1_DIS_s007
                                 + 1.16752 w_k1_FDX_s007
                                 + 1.61581 w_k1_IBM_s007
                                 + 1.29417 w_k1_INTC_s007
                                 + 1.66142 w_k1_KO_s007 + 1.89291 w_k1_MAT_s007
                                 + 1.25709 w_k1_MSFT_s007
                                 + 1.33612 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.836156 w_k1_ADM_s007
                                 + 0.606805 w_k1_AXP_s007
                                 + 0.689933 w_k1_DIS_s007
                                 + 0.844307 w_k1_FDX_s007
                                 + 0.640649 w_k1_IBM_s007
                                 + 1.00568 w_k1_INTC_s007
                                 + 0.890426 w_k1_KO_s007
                                 + 0.69032 w_k1_MAT_s007
                                 + 0.817278 w_k1_MSFT_s007
                                 + 0.864186 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.707043 w_k1_ADM_s007
                                 + 0.788369 w_k1_AXP_s007
                                 + 0.613262 w_k1_DIS_s007
                                 + 0.750259 w_k1_FDX_s007
                                 + 0.916501 w_k1_IBM_s007
                                 + 0.896037 w_k1_INTC_s007
                                 + 0.921327 w_k1_KO_s007
                                 + 0.869491 w_k1_MAT_s007
                                 + 0.850341 w_k1_MSFT_s007
                                 + 0.984768 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.903756 w_k1_ADM_s007
                                 + 0.62547 w_k1_AXP_s007
                                 + 0.886137 w_k1_DIS_s007
                                 + 0.834372 w_k1_FDX_s007
                                 + 0.879564 w_k1_IBM_s007
                                 + 0.769732 w_k1_INTC_s007
                                 + 1.0501 w_k1_KO_s007 + 0.847471 w_k1_MAT_s007
                                 + 0.909143 w_k1_MSFT_s007
                                 + 0.953057 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.19384 w_k1_ADM_s007
                                 + 1.47025 w_k1_AXP_s007
                                 + 1.73424 w_k1_DIS_s007
                                 + 1.52487 w_k1_FDX_s007
                                 + 0.882013 w_k1_IBM_s007
                                 + 1.17737 w_k1_INTC_s007
                                 + 1.30061 w_k1_KO_s007
                                 + 0.922726 w_k1_MAT_s007
                                 + 1.25337 w_k1_MSFT_s007
                                 + 1.09582 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.912458 w_k1_ADM_s010
                                 + 0.520766 w_k1_AXP_s010
                                 + 0.627334 w_k1_DIS_s010
                                 + 0.686656 w_k1_FDX_s010
                                 + 0.654837 w_k1_IBM_s010
                                 + 0.655161 w_k1_INTC_s010
                                 + 0.679396 w_k1_KO_s010
                                 + 0.531039 w_k1_MAT_s010
                                 + 0.638877 w_k1_MSFT_s010
                                 + 0.724011 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.14212 w_k1_ADM_s010
                                 + 0.736509 w_k1_AXP_s010
                                 + 0.699879 w_k1_DIS_s010
                                 + 0.766599 w_k1_FDX_s010
                                 + 1.23081 w_k1_IBM_s010
                                 + 0.884865 w_k1_INTC_s010
                                 + 0.955458 w_k1_KO_s010
                                 + 1.17016 w_k1_MAT_s010
                                 + 0.805519 w_k1_MSFT_s010
                                 + 0.934781 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.23047 w_k1_ADM_s010
                                 + 1.32986 w_k1_AXP_s010
                                 + 1.05096 w_k1_DIS_s010
                                 + 1.00931 w_k1_FDX_s010
                                 + 1.51937 w_k1_IBM_s010
                                 + 0.93174 w_k1_INTC_s010
                                 + 1.15284 w_k1_KO_s010 + 1.63779 w_k1_MAT_s010
                                 + 1.05495 w_k1_MSFT_s010
                                 + 1.10987 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.12351 w_k1_ADM_s010
                                 + 1.10419 w_k1_AXP_s010
                                 + 1.10401 w_k1_DIS_s010
                                 + 1.03557 w_k1_FDX_s010
                                 + 1.17757 w_k1_IBM_s010
                                 + 0.942291 w_k1_INTC_s010
                                 + 1.17591 w_k1_KO_s010 + 1.50087 w_k1_MAT_s010
                                 + 1.07758 w_k1_MSFT_s010
                                 + 0.948285 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.73768 w_k1_ADM_s010
                                 + 1.63269 w_k1_AXP_s010
                                 + 1.60451 w_k1_DIS_s010
                                 + 1.60359 w_k1_FDX_s010 + 1.1041 w_k1_IBM_s010
                                 + 1.15372 w_k1_INTC_s010
                                 + 1.24027 w_k1_KO_s010 + 1.61427 w_k1_MAT_s010
                                 + 1.43197 w_k1_MSFT_s010
                                 + 1.20522 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.825292 w_k1_ADM_s010
                                 + 0.783554 w_k1_AXP_s010
                                 + 0.735644 w_k1_DIS_s010
                                 + 0.757838 w_k1_FDX_s010
                                 + 0.916357 w_k1_IBM_s010
                                 + 0.714846 w_k1_INTC_s010
                                 + 0.727932 w_k1_KO_s010
                                 + 1.02397 w_k1_MAT_s010
                                 + 0.732698 w_k1_MSFT_s010
                                 + 0.854665 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.35746 w_k1_ADM_s010
                                 + 1.23893 w_k1_AXP_s010
                                 + 1.30625 w_k1_DIS_s010
                                 + 1.27708 w_k1_FDX_s010
                                 + 1.13142 w_k1_IBM_s010
                                 + 1.14142 w_k1_INTC_s010
                                 + 1.13947 w_k1_KO_s010 + 1.12813 w_k1_MAT_s010
                                 + 1.11986 w_k1_MSFT_s010
                                 + 1.14958 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.38076 w_k1_ADM_s010
                                 + 0.840299 w_k1_AXP_s010
                                 + 0.771599 w_k1_DIS_s010
                                 + 0.96654 w_k1_FDX_s010
                                 + 0.753371 w_k1_IBM_s010
                                 + 0.78901 w_k1_INTC_s010
                                 + 0.778118 w_k1_KO_s010
                                 + 0.863503 w_k1_MAT_s010
                                 + 0.868222 w_k1_MSFT_s010
                                 + 0.986191 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.825086 w_k1_ADM_s011
                                 + 0.956666 w_k1_AXP_s011
                                 + 1.01437 w_k1_DIS_s011
                                 + 0.878187 w_k1_FDX_s011
                                 + 0.895156 w_k1_IBM_s011
                                 + 1.09082 w_k1_INTC_s011
                                 + 0.940275 w_k1_KO_s011
                                 + 0.737994 w_k1_MAT_s011
                                 + 0.969745 w_k1_MSFT_s011
                                 + 0.924986 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.33553 w_k1_ADM_s011
                                 + 1.72391 w_k1_AXP_s011
                                 + 1.70632 w_k1_DIS_s011
                                 + 1.18472 w_k1_FDX_s011
                                 + 1.70022 w_k1_IBM_s011
                                 + 1.22994 w_k1_INTC_s011
                                 + 1.47243 w_k1_KO_s011 + 2.21586 w_k1_MAT_s011
                                 + 1.32251 w_k1_MSFT_s011
                                 + 1.27753 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.926751 w_k1_ADM_s011
                                 + 0.576456 w_k1_AXP_s011
                                 + 0.571525 w_k1_DIS_s011
                                 + 0.767839 w_k1_FDX_s011
                                 + 0.596077 w_k1_IBM_s011
                                 + 0.880259 w_k1_INTC_s011
                                 + 0.611333 w_k1_KO_s011
                                 + 0.509865 w_k1_MAT_s011
                                 + 0.767368 w_k1_MSFT_s011
                                 + 0.710441 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.15536 w_k1_ADM_s011
                                 + 1.17035 w_k1_AXP_s011
                                 + 1.05996 w_k1_DIS_s011
                                 + 1.10223 w_k1_FDX_s011
                                 + 0.976573 w_k1_IBM_s011
                                 + 1.32817 w_k1_INTC_s011
                                 + 0.942724 w_k1_KO_s011
                                 + 0.994115 w_k1_MAT_s011
                                 + 1.17974 w_k1_MSFT_s011
                                 + 0.903005 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.722512 w_k1_ADM_s011
                                 + 0.878364 w_k1_AXP_s011
                                 + 0.813342 w_k1_DIS_s011
                                 + 0.758305 w_k1_FDX_s011
                                 + 0.978297 w_k1_IBM_s011
                                 + 1.07338 w_k1_INTC_s011
                                 + 0.786617 w_k1_KO_s011
                                 + 0.980576 w_k1_MAT_s011
                                 + 0.929511 w_k1_MSFT_s011
                                 + 0.751465 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.966884 w_k1_ADM_s011
                                 + 1.06749 w_k1_AXP_s011
                                 + 1.01585 w_k1_DIS_s011
                                 + 0.93073 w_k1_FDX_s011
                                 + 1.27902 w_k1_IBM_s011
                                 + 1.23233 w_k1_INTC_s011
                                 + 1.09346 w_k1_KO_s011 + 1.33063 w_k1_MAT_s011
                                 + 0.931373 w_k1_MSFT_s011
                                 + 1.12083 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.21476 w_k1_ADM_s011
                                 + 1.47113 w_k1_AXP_s011
                                 + 1.60825 w_k1_DIS_s011
                                 + 1.32144 w_k1_FDX_s011
                                 + 1.06878 w_k1_IBM_s011
                                 + 1.48484 w_k1_INTC_s011
                                 + 1.17873 w_k1_KO_s011 + 1.03155 w_k1_MAT_s011
                                 + 1.37279 w_k1_MSFT_s011
                                 + 1.02561 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.89035 w_k1_ADM_s011
                                 + 0.959246 w_k1_AXP_s011
                                 + 0.915475 w_k1_DIS_s011
                                 + 1.04143 w_k1_FDX_s011
                                 + 0.714093 w_k1_IBM_s011
                                 + 1.19963 w_k1_INTC_s011
                                 + 0.660736 w_k1_KO_s011
                                 + 0.690373 w_k1_MAT_s011
                                 + 1.05663 w_k1_MSFT_s011
                                 + 0.847097 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.727071 w_k1_ADM_s012
                                 + 0.742946 w_k1_AXP_s012
                                 + 0.802609 w_k1_DIS_s012
                                 + 0.765072 w_k1_FDX_s012
                                 + 1.10193 w_k1_IBM_s012
                                 + 0.983544 w_k1_INTC_s012
                                 + 1.01166 w_k1_KO_s012
                                 + 0.875504 w_k1_MAT_s012
                                 + 0.913247 w_k1_MSFT_s012
                                 + 0.819107 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.837633 w_k1_ADM_s012
                                 + 0.872439 w_k1_AXP_s012
                                 + 0.875828 w_k1_DIS_s012
                                 + 0.891352 w_k1_FDX_s012
                                 + 0.904984 w_k1_IBM_s012
                                 + 0.791969 w_k1_INTC_s012
                                 + 0.984784 w_k1_KO_s012
                                 + 0.960368 w_k1_MAT_s012
                                 + 0.819421 w_k1_MSFT_s012
                                 + 0.940512 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.21483 w_k1_ADM_s012
                                 + 1.47744 w_k1_AXP_s012
                                 + 1.26935 w_k1_DIS_s012
                                 + 1.32944 w_k1_FDX_s012
                                 + 1.07951 w_k1_IBM_s012
                                 + 1.00835 w_k1_INTC_s012
                                 + 1.02866 w_k1_KO_s012 + 1.23672 w_k1_MAT_s012
                                 + 1.11881 w_k1_MSFT_s012
                                 + 1.09502 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.949529 w_k1_ADM_s012
                                 + 1.11331 w_k1_AXP_s012 + 1.0156 w_k1_DIS_s012
                                 + 0.995686 w_k1_FDX_s012
                                 + 1.16575 w_k1_IBM_s012
                                 + 0.846709 w_k1_INTC_s012
                                 + 1.01559 w_k1_KO_s012 + 1.53225 w_k1_MAT_s012
                                 + 0.892719 w_k1_MSFT_s012
                                 + 0.970678 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.849025 w_k1_ADM_s012
                                 + 0.978638 w_k1_AXP_s012
                                 + 0.724471 w_k1_DIS_s012
                                 + 0.834147 w_k1_FDX_s012
                                 + 1.03615 w_k1_IBM_s012
                                 + 0.795258 w_k1_INTC_s012
                                 + 0.790338 w_k1_KO_s012
                                 + 1.09083 w_k1_MAT_s012
                                 + 0.762687 w_k1_MSFT_s012
                                 + 0.838572 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.11529 w_k1_ADM_s012
                                 + 1.29168 w_k1_AXP_s012
                                 + 1.36591 w_k1_DIS_s012
                                 + 1.17265 w_k1_FDX_s012 + 1.4195 w_k1_IBM_s012
                                 + 1.09228 w_k1_INTC_s012
                                 + 1.38581 w_k1_KO_s012 + 1.63578 w_k1_MAT_s012
                                 + 1.04399 w_k1_MSFT_s012
                                 + 1.10362 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.0594 w_k1_ADM_s012
                                 + 0.922452 w_k1_AXP_s012
                                 + 1.23094 w_k1_DIS_s012
                                 + 1.17533 w_k1_FDX_s012
                                 + 0.861347 w_k1_IBM_s012
                                 + 1.00157 w_k1_INTC_s012
                                 + 1.14221 w_k1_KO_s012 + 1.00453 w_k1_MAT_s012
                                 + 1.00949 w_k1_MSFT_s012
                                 + 1.08307 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.987347 w_k1_ADM_s012
                                 + 0.756862 w_k1_AXP_s012
                                 + 0.827466 w_k1_DIS_s012
                                 + 0.863093 w_k1_FDX_s012
                                 + 0.677298 w_k1_IBM_s012
                                 + 0.679194 w_k1_INTC_s012
                                 + 0.774116 w_k1_KO_s012
                                 + 0.639505 w_k1_MAT_s012
                                 + 0.79351 w_k1_MSFT_s012
                                 + 0.831016 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.42408 w_k1_ADM_s013
                                 + 1.58677 w_k1_AXP_s013
                                 + 1.41127 w_k1_DIS_s013
                                 + 1.26117 w_k1_FDX_s013 + 1.0942 w_k1_IBM_s013
                                 + 1.18288 w_k1_INTC_s013
                                 + 1.09901 w_k1_KO_s013 + 1.08854 w_k1_MAT_s013
                                 + 1.19599 w_k1_MSFT_s013 + 1.1461 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.14746 w_k1_ADM_s013
                                 + 1.09346 w_k1_AXP_s013
                                 + 1.01219 w_k1_DIS_s013
                                 + 1.05677 w_k1_FDX_s013
                                 + 0.970384 w_k1_IBM_s013
                                 + 1.13553 w_k1_INTC_s013
                                 + 0.953399 w_k1_KO_s013
                                 + 0.859393 w_k1_MAT_s013
                                 + 1.02772 w_k1_MSFT_s013 + 1.0097 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.63876 w_k1_ADM_s013
                                 + 1.0278 w_k1_AXP_s013 + 1.3756 w_k1_DIS_s013
                                 + 1.22413 w_k1_FDX_s013
                                 + 0.792601 w_k1_IBM_s013
                                 + 0.967419 w_k1_INTC_s013
                                 + 1.02582 w_k1_KO_s013
                                 + 0.675935 w_k1_MAT_s013
                                 + 1.15426 w_k1_MSFT_s013
                                 + 1.06298 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.13935 w_k1_ADM_s013
                                 + 1.1694 w_k1_AXP_s013 + 1.33076 w_k1_DIS_s013
                                 + 1.02418 w_k1_FDX_s013
                                 + 1.34371 w_k1_IBM_s013
                                 + 1.11475 w_k1_INTC_s013
                                 + 1.30907 w_k1_KO_s013 + 1.32451 w_k1_MAT_s013
                                 + 1.03707 w_k1_MSFT_s013 + 1.0736 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.26878 w_k1_ADM_s013
                                 + 1.34697 w_k1_AXP_s013
                                 + 1.35235 w_k1_DIS_s013
                                 + 1.02881 w_k1_FDX_s013
                                 + 1.96728 w_k1_IBM_s013
                                 + 1.48561 w_k1_INTC_s013
                                 + 1.55508 w_k1_KO_s013 + 2.04091 w_k1_MAT_s013
                                 + 0.985157 w_k1_MSFT_s013
                                 + 1.3159 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.15765 w_k1_ADM_s013
                                 + 1.18457 w_k1_AXP_s013
                                 + 0.635232 w_k1_DIS_s013
                                 + 0.944834 w_k1_FDX_s013
                                 + 0.818889 w_k1_IBM_s013
                                 + 0.740475 w_k1_INTC_s013
                                 + 0.730506 w_k1_KO_s013
                                 + 1.29117 w_k1_MAT_s013
                                 + 0.797657 w_k1_MSFT_s013
                                 + 0.98489 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.814326 w_k1_ADM_s013
                                 + 0.812314 w_k1_AXP_s013
                                 + 0.823495 w_k1_DIS_s013
                                 + 0.818916 w_k1_FDX_s013
                                 + 1.01377 w_k1_IBM_s013
                                 + 0.7757 w_k1_INTC_s013
                                 + 0.941019 w_k1_KO_s013
                                 + 1.01811 w_k1_MAT_s013
                                 + 0.851111 w_k1_MSFT_s013
                                 + 0.923821 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.827348 w_k1_ADM_s013
                                 + 0.786017 w_k1_AXP_s013
                                 + 0.862729 w_k1_DIS_s013
                                 + 0.864945 w_k1_FDX_s013
                                 + 0.609194 w_k1_IBM_s013
                                 + 0.847928 w_k1_INTC_s013
                                 + 0.759652 w_k1_KO_s013
                                 + 0.58888 w_k1_MAT_s013
                                 + 0.833305 w_k1_MSFT_s013
                                 + 0.870073 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.72082 w_k1_ADM_s014
                                 + 0.752924 w_k1_AXP_s014
                                 + 0.686044 w_k1_DIS_s014
                                 + 0.732109 w_k1_FDX_s014
                                 + 0.963854 w_k1_IBM_s014
                                 + 0.855552 w_k1_INTC_s014
                                 + 0.927864 w_k1_KO_s014
                                 + 1.03953 w_k1_MAT_s014
                                 + 0.84626 w_k1_MSFT_s014
                                 + 0.842037 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.50451 w_k1_ADM_s014
                                 + 1.28139 w_k1_AXP_s014
                                 + 1.43687 w_k1_DIS_s014
                                 + 1.29321 w_k1_FDX_s014
                                 + 1.26892 w_k1_IBM_s014
                                 + 1.00062 w_k1_INTC_s014
                                 + 1.35393 w_k1_KO_s014 + 1.68521 w_k1_MAT_s014
                                 + 1.24004 w_k1_MSFT_s014
                                 + 1.11419 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.18424 w_k1_ADM_s014
                                 + 1.38228 w_k1_AXP_s014
                                 + 1.22276 w_k1_DIS_s014
                                 + 1.22779 w_k1_FDX_s014
                                 + 1.12285 w_k1_IBM_s014
                                 + 0.995224 w_k1_INTC_s014
                                 + 0.989776 w_k1_KO_s014
                                 + 1.82338 w_k1_MAT_s014
                                 + 1.08721 w_k1_MSFT_s014
                                 + 1.08565 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.00147 w_k1_ADM_s014
                                 + 0.939121 w_k1_AXP_s014
                                 + 0.994225 w_k1_DIS_s014
                                 + 1.06494 w_k1_FDX_s014
                                 + 1.08882 w_k1_IBM_s014
                                 + 0.965324 w_k1_INTC_s014
                                 + 0.90535 w_k1_KO_s014 + 1.06391 w_k1_MAT_s014
                                 + 1.16884 w_k1_MSFT_s014
                                 + 0.968103 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.709746 w_k1_ADM_s014
                                 + 0.599084 w_k1_AXP_s014
                                 + 0.562074 w_k1_DIS_s014
                                 + 0.753954 w_k1_FDX_s014
                                 + 0.68426 w_k1_IBM_s014
                                 + 0.677407 w_k1_INTC_s014
                                 + 0.607285 w_k1_KO_s014
                                 + 0.716617 w_k1_MAT_s014
                                 + 0.727629 w_k1_MSFT_s014
                                 + 0.716753 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.856226 w_k1_ADM_s014
                                 + 0.699012 w_k1_AXP_s014
                                 + 0.884276 w_k1_DIS_s014
                                 + 0.858498 w_k1_FDX_s014
                                 + 1.29096 w_k1_IBM_s014
                                 + 0.997579 w_k1_INTC_s014
                                 + 1.27231 w_k1_KO_s014 + 1.16951 w_k1_MAT_s014
                                 + 0.98 w_k1_MSFT_s014 + 0.882947 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.96333 w_k1_ADM_s014
                                 + 0.835485 w_k1_AXP_s014
                                 + 0.837332 w_k1_DIS_s014
                                 + 1.00948 w_k1_FDX_s014
                                 + 0.805062 w_k1_IBM_s014
                                 + 1.04562 w_k1_INTC_s014
                                 + 0.825797 w_k1_KO_s014
                                 + 0.930074 w_k1_MAT_s014
                                 + 1.03693 w_k1_MSFT_s014
                                 + 0.874767 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.959764 w_k1_ADM_s014
                                 + 1.21131 w_k1_AXP_s014
                                 + 0.862644 w_k1_DIS_s014
                                 + 0.858709 w_k1_FDX_s014
                                 + 1.25941 w_k1_IBM_s014
                                 + 0.803028 w_k1_INTC_s014
                                 + 1.05326 w_k1_KO_s014 + 1.66178 w_k1_MAT_s014
                                 + 1.02168 w_k1_MSFT_s014
                                 + 0.935878 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.09357 w_k1_ADM_s015
                                 + 1.06522 w_k1_AXP_s015
                                 + 1.06456 w_k1_DIS_s015 + 1.0582 w_k1_FDX_s015
                                 + 1.41326 w_k1_IBM_s015
                                 + 1.3247 w_k1_INTC_s015 + 1.26389 w_k1_KO_s015
                                 + 0.993686 w_k1_MAT_s015
                                 + 1.27041 w_k1_MSFT_s015
                                 + 1.17721 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.796879 w_k1_ADM_s015
                                 + 0.617802 w_k1_AXP_s015
                                 + 0.562975 w_k1_DIS_s015
                                 + 0.755555 w_k1_FDX_s015
                                 + 1.04144 w_k1_IBM_s015
                                 + 0.830776 w_k1_INTC_s015
                                 + 0.789613 w_k1_KO_s015
                                 + 0.823738 w_k1_MAT_s015
                                 + 0.76356 w_k1_MSFT_s015
                                 + 0.847669 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.16548 w_k1_ADM_s015
                                 + 0.934265 w_k1_AXP_s015
                                 + 0.958116 w_k1_DIS_s015
                                 + 1.04628 w_k1_FDX_s015
                                 + 0.764125 w_k1_IBM_s015
                                 + 1.26635 w_k1_INTC_s015
                                 + 0.937268 w_k1_KO_s015
                                 + 0.679056 w_k1_MAT_s015
                                 + 1.05736 w_k1_MSFT_s015
                                 + 0.935211 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.96931 w_k1_ADM_s015
                                 + 0.793063 w_k1_AXP_s015
                                 + 0.804172 w_k1_DIS_s015
                                 + 0.934135 w_k1_FDX_s015
                                 + 0.932722 w_k1_IBM_s015
                                 + 1.01277 w_k1_INTC_s015
                                 + 0.961859 w_k1_KO_s015
                                 + 0.695152 w_k1_MAT_s015
                                 + 0.961144 w_k1_MSFT_s015
                                 + 0.88111 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.32449 w_k1_ADM_s015
                                 + 1.33019 w_k1_AXP_s015
                                 + 1.44529 w_k1_DIS_s015
                                 + 1.38761 w_k1_FDX_s015
                                 + 0.963981 w_k1_IBM_s015
                                 + 1.31611 w_k1_INTC_s015
                                 + 1.21649 w_k1_KO_s015 + 1.00797 w_k1_MAT_s015
                                 + 1.50696 w_k1_MSFT_s015
                                 + 1.20601 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.846449 w_k1_ADM_s015
                                 + 0.692149 w_k1_AXP_s015
                                 + 0.665967 w_k1_DIS_s015
                                 + 0.845651 w_k1_FDX_s015
                                 + 0.636026 w_k1_IBM_s015
                                 + 0.849601 w_k1_INTC_s015
                                 + 0.637525 w_k1_KO_s015
                                 + 0.453459 w_k1_MAT_s015
                                 + 0.809514 w_k1_MSFT_s015
                                 + 0.839167 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.35205 w_k1_ADM_s015
                                 + 1.01792 w_k1_AXP_s015
                                 + 1.20737 w_k1_DIS_s015
                                 + 1.10282 w_k1_FDX_s015
                                 + 0.960255 w_k1_IBM_s015
                                 + 1.27594 w_k1_INTC_s015
                                 + 1.27992 w_k1_KO_s015
                                 + 0.777743 w_k1_MAT_s015
                                 + 1.18255 w_k1_MSFT_s015
                                 + 1.06356 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.11879 w_k1_ADM_s015
                                 + 1.14179 w_k1_AXP_s015
                                 + 1.02433 w_k1_DIS_s015
                                 + 1.09868 w_k1_FDX_s015 + 1.1224 w_k1_IBM_s015
                                 + 1.2099 w_k1_INTC_s015 + 1.03122 w_k1_KO_s015
                                 + 1.05896 w_k1_MAT_s015
                                 + 1.14436 w_k1_MSFT_s015
                                 + 0.888218 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.892277 w_k1_ADM_s016
                                 + 0.728794 w_k1_AXP_s016
                                 + 0.679273 w_k1_DIS_s016
                                 + 0.722505 w_k1_FDX_s016
                                 + 0.767125 w_k1_IBM_s016
                                 + 0.651705 w_k1_INTC_s016
                                 + 0.738378 w_k1_KO_s016
                                 + 0.707714 w_k1_MAT_s016
                                 + 0.668997 w_k1_MSFT_s016
                                 + 0.820265 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.15465 w_k1_ADM_s016
                                 + 0.899125 w_k1_AXP_s016
                                 + 0.915743 w_k1_DIS_s016
                                 + 0.859106 w_k1_FDX_s016
                                 + 1.12988 w_k1_IBM_s016
                                 + 0.936095 w_k1_INTC_s016
                                 + 0.986556 w_k1_KO_s016
                                 + 1.10847 w_k1_MAT_s016
                                 + 0.822887 w_k1_MSFT_s016
                                 + 1.01543 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.52139 w_k1_ADM_s016
                                 + 1.06144 w_k1_AXP_s016
                                 + 1.28702 w_k1_DIS_s016 + 1.1494 w_k1_FDX_s016
                                 + 0.881044 w_k1_IBM_s016
                                 + 1.01663 w_k1_INTC_s016 + 1.1797 w_k1_KO_s016
                                 + 0.805946 w_k1_MAT_s016
                                 + 1.09355 w_k1_MSFT_s016
                                 + 1.00268 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.401 w_k1_ADM_s016
                                 + 1.40278 w_k1_AXP_s016
                                 + 1.30773 w_k1_DIS_s016
                                 + 1.17583 w_k1_FDX_s016
                                 + 1.41973 w_k1_IBM_s016
                                 + 1.04215 w_k1_INTC_s016 + 1.4573 w_k1_KO_s016
                                 + 1.78752 w_k1_MAT_s016
                                 + 1.08993 w_k1_MSFT_s016
                                 + 1.25348 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.51171 w_k1_ADM_s016
                                 + 1.38871 w_k1_AXP_s016
                                 + 1.12578 w_k1_DIS_s016
                                 + 1.23797 w_k1_FDX_s016
                                 + 0.974206 w_k1_IBM_s016
                                 + 1.21827 w_k1_INTC_s016
                                 + 1.04604 w_k1_KO_s016 + 1.20665 w_k1_MAT_s016
                                 + 1.17588 w_k1_MSFT_s016 + 1.0317 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.11034 w_k1_ADM_s016
                                 + 1.16145 w_k1_AXP_s016
                                 + 1.00772 w_k1_DIS_s016
                                 + 0.982187 w_k1_FDX_s016
                                 + 1.71781 w_k1_IBM_s016
                                 + 0.826182 w_k1_INTC_s016
                                 + 1.19131 w_k1_KO_s016 + 1.57538 w_k1_MAT_s016
                                 + 0.844386 w_k1_MSFT_s016
                                 + 1.07019 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.29031 w_k1_ADM_s016
                                 + 0.856811 w_k1_AXP_s016
                                 + 0.8646 w_k1_DIS_s016
                                 + 0.955406 w_k1_FDX_s016
                                 + 0.60935 w_k1_IBM_s016
                                 + 0.894429 w_k1_INTC_s016
                                 + 0.739416 w_k1_KO_s016
                                 + 0.553372 w_k1_MAT_s016
                                 + 0.80791 w_k1_MSFT_s016
                                 + 0.864138 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.33249 w_k1_ADM_s016
                                 + 1.20131 w_k1_AXP_s016
                                 + 0.865843 w_k1_DIS_s016
                                 + 1.0003 w_k1_FDX_s016
                                 + 0.824801 w_k1_IBM_s016
                                 + 1.07519 w_k1_INTC_s016
                                 + 0.858307 w_k1_KO_s016
                                 + 0.823672 w_k1_MAT_s016
                                 + 0.960785 w_k1_MSFT_s016
                                 + 0.936681 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.793192 w_k1_ADM_s017
                                 + 0.861045 w_k1_AXP_s017
                                 + 0.693052 w_k1_DIS_s017
                                 + 0.843751 w_k1_FDX_s017
                                 + 0.876977 w_k1_IBM_s017
                                 + 0.761279 w_k1_INTC_s017
                                 + 0.72431 w_k1_KO_s017 + 1.07645 w_k1_MAT_s017
                                 + 0.696703 w_k1_MSFT_s017
                                 + 0.887291 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.858889 w_k1_ADM_s017
                                 + 0.925253 w_k1_AXP_s017
                                 + 0.815763 w_k1_DIS_s017
                                 + 0.969988 w_k1_FDX_s017
                                 + 0.687102 w_k1_IBM_s017
                                 + 0.945882 w_k1_INTC_s017
                                 + 0.703712 w_k1_KO_s017
                                 + 0.702902 w_k1_MAT_s017
                                 + 0.848938 w_k1_MSFT_s017
                                 + 0.917918 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.942207 w_k1_ADM_s017
                                 + 0.773285 w_k1_AXP_s017
                                 + 1.1897 w_k1_DIS_s017 + 1.06529 w_k1_FDX_s017
                                 + 0.724989 w_k1_IBM_s017
                                 + 1.07689 w_k1_INTC_s017
                                 + 0.808081 w_k1_KO_s017
                                 + 0.637784 w_k1_MAT_s017
                                 + 1.00913 w_k1_MSFT_s017
                                 + 0.974442 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.09634 w_k1_ADM_s017
                                 + 1.43644 w_k1_AXP_s017
                                 + 1.04969 w_k1_DIS_s017
                                 + 1.27302 w_k1_FDX_s017
                                 + 0.956536 w_k1_IBM_s017
                                 + 1.19595 w_k1_INTC_s017
                                 + 0.803327 w_k1_KO_s017
                                 + 1.11077 w_k1_MAT_s017
                                 + 1.01065 w_k1_MSFT_s017
                                 + 1.04359 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.961927 w_k1_ADM_s017
                                 + 1.28485 w_k1_AXP_s017 + 1.1256 w_k1_DIS_s017
                                 + 1.08603 w_k1_FDX_s017
                                 + 1.26386 w_k1_IBM_s017
                                 + 1.16127 w_k1_INTC_s017
                                 + 1.12127 w_k1_KO_s017 + 1.467 w_k1_MAT_s017
                                 + 0.942838 w_k1_MSFT_s017
                                 + 1.13202 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.828107 w_k1_ADM_s017
                                 + 0.90261 w_k1_AXP_s017
                                 + 1.24146 w_k1_DIS_s017
                                 + 1.12869 w_k1_FDX_s017
                                 + 0.987875 w_k1_IBM_s017
                                 + 0.998815 w_k1_INTC_s017
                                 + 0.992076 w_k1_KO_s017
                                 + 1.07589 w_k1_MAT_s017
                                 + 0.954217 w_k1_MSFT_s017
                                 + 1.02133 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.671706 w_k1_ADM_s017
                                 + 0.586008 w_k1_AXP_s017
                                 + 0.677025 w_k1_DIS_s017
                                 + 0.749607 w_k1_FDX_s017
                                 + 0.703662 w_k1_IBM_s017
                                 + 0.830318 w_k1_INTC_s017
                                 + 0.674018 w_k1_KO_s017
                                 + 0.554828 w_k1_MAT_s017
                                 + 0.683776 w_k1_MSFT_s017
                                 + 0.853547 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.741365 w_k1_ADM_s017
                                 + 0.916187 w_k1_AXP_s017
                                 + 0.901669 w_k1_DIS_s017
                                 + 0.883034 w_k1_FDX_s017
                                 + 1.22913 w_k1_IBM_s017
                                 + 0.966491 w_k1_INTC_s017
                                 + 0.924749 w_k1_KO_s017 + 1.4454 w_k1_MAT_s017
                                 + 0.803758 w_k1_MSFT_s017
                                 + 1.00972 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.938943 w_k1_ADM_s100
                                 + 0.856348 w_k1_AXP_s100
                                 + 0.945636 w_k1_DIS_s100
                                 + 0.854514 w_k1_FDX_s100
                                 + 1.29122 w_k1_IBM_s100
                                 + 1.02497 w_k1_INTC_s100
                                 + 1.22475 w_k1_KO_s100
                                 + 0.879989 w_k1_MAT_s100
                                 + 0.871253 w_k1_MSFT_s100
                                 + 0.967346 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.832397 w_k1_ADM_s100
                                 + 0.531244 w_k1_AXP_s100
                                 + 0.555421 w_k1_DIS_s100
                                 + 0.700176 w_k1_FDX_s100
                                 + 0.777438 w_k1_IBM_s100
                                 + 0.710348 w_k1_INTC_s100
                                 + 0.833858 w_k1_KO_s100
                                 + 0.705535 w_k1_MAT_s100
                                 + 0.732375 w_k1_MSFT_s100
                                 + 0.818537 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.07157 w_k1_ADM_s100
                                 + 1.05206 w_k1_AXP_s100
                                 + 1.07807 w_k1_DIS_s100
                                 + 1.09022 w_k1_FDX_s100
                                 + 1.03857 w_k1_IBM_s100
                                 + 1.19552 w_k1_INTC_s100
                                 + 1.13745 w_k1_KO_s100
                                 + 0.599818 w_k1_MAT_s100
                                 + 1.16029 w_k1_MSFT_s100
                                 + 1.12515 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.09334 w_k1_ADM_s100
                                 + 1.0739 w_k1_AXP_s100
                                 + 0.775608 w_k1_DIS_s100
                                 + 0.920086 w_k1_FDX_s100
                                 + 1.25367 w_k1_IBM_s100
                                 + 0.875014 w_k1_INTC_s100
                                 + 1.05699 w_k1_KO_s100 + 1.47179 w_k1_MAT_s100
                                 + 0.937461 w_k1_MSFT_s100
                                 + 1.03395 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.14146 w_k1_ADM_s100
                                 + 1.39187 w_k1_AXP_s100
                                 + 1.19699 w_k1_DIS_s100
                                 + 1.10874 w_k1_FDX_s100
                                 + 1.65555 w_k1_IBM_s100
                                 + 1.23921 w_k1_INTC_s100 + 1.3295 w_k1_KO_s100
                                 + 1.30543 w_k1_MAT_s100
                                 + 0.98791 w_k1_MSFT_s100
                                 + 1.04044 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.940155 w_k1_ADM_s100
                                 + 0.752237 w_k1_AXP_s100
                                 + 0.772842 w_k1_DIS_s100
                                 + 0.893181 w_k1_FDX_s100
                                 + 0.778196 w_k1_IBM_s100
                                 + 1.05772 w_k1_INTC_s100
                                 + 0.915099 w_k1_KO_s100
                                 + 0.520996 w_k1_MAT_s100
                                 + 0.980535 w_k1_MSFT_s100
                                 + 0.904099 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.22271 w_k1_ADM_s100
                                 + 1.40935 w_k1_AXP_s100
                                 + 1.46671 w_k1_DIS_s100
                                 + 1.44914 w_k1_FDX_s100
                                 + 1.25449 w_k1_IBM_s100
                                 + 1.43809 w_k1_INTC_s100
                                 + 1.40884 w_k1_KO_s100
                                 + 0.967383 w_k1_MAT_s100
                                 + 1.28883 w_k1_MSFT_s100
                                 + 1.34283 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.37325 w_k1_ADM_s100
                                 + 1.02272 w_k1_AXP_s100
                                 + 1.22286 w_k1_DIS_s100
                                 + 1.26781 w_k1_FDX_s100
                                 + 0.785223 w_k1_IBM_s100
                                 + 1.09884 w_k1_INTC_s100
                                 + 0.938844 w_k1_KO_s100
                                 + 0.512003 w_k1_MAT_s100
                                 + 1.17607 w_k1_MSFT_s100 + 1.0401 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.942086 w_k1_ADM_s101
                                 + 0.970606 w_k1_AXP_s101
                                 + 0.853541 w_k1_DIS_s101
                                 + 0.852163 w_k1_FDX_s101
                                 + 0.856557 w_k1_IBM_s101
                                 + 0.851082 w_k1_INTC_s101
                                 + 0.932583 w_k1_KO_s101 + 1.1192 w_k1_MAT_s101
                                 + 0.9083 w_k1_MSFT_s101
                                 + 0.935682 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.964127 w_k1_ADM_s101
                                 + 0.993992 w_k1_AXP_s101
                                 + 0.804648 w_k1_DIS_s101
                                 + 0.797528 w_k1_FDX_s101
                                 + 1.34653 w_k1_IBM_s101
                                 + 1.09664 w_k1_INTC_s101
                                 + 1.07147 w_k1_KO_s101 + 1.37462 w_k1_MAT_s101
                                 + 1.0243 w_k1_MSFT_s101
                                 + 0.991411 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12002 w_k1_ADM_s101
                                 + 1.02159 w_k1_AXP_s101
                                 + 1.19566 w_k1_DIS_s101
                                 + 1.06445 w_k1_FDX_s101
                                 + 1.17817 w_k1_IBM_s101
                                 + 0.95675 w_k1_INTC_s101
                                 + 1.21061 w_k1_KO_s101 + 1.06624 w_k1_MAT_s101
                                 + 0.900965 w_k1_MSFT_s101
                                 + 1.12446 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.06971 w_k1_ADM_s101
                                 + 1.38346 w_k1_AXP_s101
                                 + 1.58313 w_k1_DIS_s101
                                 + 1.42707 w_k1_FDX_s101
                                 + 0.899884 w_k1_IBM_s101
                                 + 1.21178 w_k1_INTC_s101
                                 + 1.06032 w_k1_KO_s101 + 1.40077 w_k1_MAT_s101
                                 + 1.37996 w_k1_MSFT_s101
                                 + 1.10058 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.707759 w_k1_ADM_s101
                                 + 0.687103 w_k1_AXP_s101
                                 + 0.651056 w_k1_DIS_s101
                                 + 0.685741 w_k1_FDX_s101
                                 + 0.731569 w_k1_IBM_s101
                                 + 0.695513 w_k1_INTC_s101
                                 + 0.767485 w_k1_KO_s101
                                 + 0.714568 w_k1_MAT_s101
                                 + 0.762401 w_k1_MSFT_s101
                                 + 0.8372 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.909247 w_k1_ADM_s101
                                 + 0.74026 w_k1_AXP_s101
                                 + 0.96158 w_k1_DIS_s101
                                 + 0.970222 w_k1_FDX_s101
                                 + 0.764827 w_k1_IBM_s101
                                 + 0.905426 w_k1_INTC_s101
                                 + 0.911306 w_k1_KO_s101
                                 + 0.664423 w_k1_MAT_s101
                                 + 0.986242 w_k1_MSFT_s101
                                 + 0.871227 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.890926 w_k1_ADM_s101
                                 + 0.708851 w_k1_AXP_s101
                                 + 0.946723 w_k1_DIS_s101
                                 + 0.886125 w_k1_FDX_s101
                                 + 1.0852 w_k1_IBM_s101
                                 + 0.966286 w_k1_INTC_s101
                                 + 1.05939 w_k1_KO_s101 + 1.07008 w_k1_MAT_s101
                                 + 1.02753 w_k1_MSFT_s101
                                 + 0.860211 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.89068 w_k1_ADM_s101
                                 + 0.935579 w_k1_AXP_s101
                                 + 1.51045 w_k1_DIS_s101
                                 + 1.16957 w_k1_FDX_s101
                                 + 0.933941 w_k1_IBM_s101
                                 + 1.0976 w_k1_INTC_s101 + 1.10836 w_k1_KO_s101
                                 + 0.783118 w_k1_MAT_s101
                                 + 1.10774 w_k1_MSFT_s101
                                 + 1.14681 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.04191 w_k1_ADM_s102
                                 + 1.28992 w_k1_AXP_s102
                                 + 1.09576 w_k1_DIS_s102
                                 + 0.989502 w_k1_FDX_s102
                                 + 1.49501 w_k1_IBM_s102
                                 + 0.973062 w_k1_INTC_s102
                                 + 1.24139 w_k1_KO_s102 + 1.59717 w_k1_MAT_s102
                                 + 1.0724 w_k1_MSFT_s102 + 1.00881 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.03506 w_k1_ADM_s102
                                 + 1.02055 w_k1_AXP_s102
                                 + 0.93077 w_k1_DIS_s102
                                 + 0.938422 w_k1_FDX_s102
                                 + 1.24533 w_k1_IBM_s102
                                 + 0.851575 w_k1_INTC_s102
                                 + 0.978208 w_k1_KO_s102
                                 + 1.10285 w_k1_MAT_s102
                                 + 0.985841 w_k1_MSFT_s102
                                 + 0.93672 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.27432 w_k1_ADM_s102
                                 + 1.85243 w_k1_AXP_s102
                                 + 1.65976 w_k1_DIS_s102
                                 + 1.41335 w_k1_FDX_s102
                                 + 1.33478 w_k1_IBM_s102
                                 + 1.28152 w_k1_INTC_s102
                                 + 1.20183 w_k1_KO_s102 + 1.77801 w_k1_MAT_s102
                                 + 1.27518 w_k1_MSFT_s102
                                 + 1.16578 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.852007 w_k1_ADM_s102
                                 + 1.00293 w_k1_AXP_s102
                                 + 0.742817 w_k1_DIS_s102
                                 + 0.809163 w_k1_FDX_s102
                                 + 0.925105 w_k1_IBM_s102
                                 + 0.904702 w_k1_INTC_s102
                                 + 0.713859 w_k1_KO_s102
                                 + 0.927201 w_k1_MAT_s102
                                 + 0.858944 w_k1_MSFT_s102
                                 + 0.888069 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.697459 w_k1_ADM_s102
                                 + 0.721475 w_k1_AXP_s102
                                 + 0.608458 w_k1_DIS_s102
                                 + 0.7918 w_k1_FDX_s102
                                 + 0.684966 w_k1_IBM_s102
                                 + 0.688264 w_k1_INTC_s102
                                 + 0.653761 w_k1_KO_s102
                                 + 1.11685 w_k1_MAT_s102
                                 + 0.812535 w_k1_MSFT_s102
                                 + 0.827479 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.03597 w_k1_ADM_s102
                                 + 1.14726 w_k1_AXP_s102
                                 + 1.10864 w_k1_DIS_s102
                                 + 1.28053 w_k1_FDX_s102 + 0.885 w_k1_IBM_s102
                                 + 1.08474 w_k1_INTC_s102
                                 + 0.907236 w_k1_KO_s102
                                 + 1.12222 w_k1_MAT_s102
                                 + 1.17558 w_k1_MSFT_s102
                                 + 1.04002 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.18757 w_k1_ADM_s102
                                 + 1.21203 w_k1_AXP_s102
                                 + 1.26452 w_k1_DIS_s102
                                 + 1.23329 w_k1_FDX_s102 + 1.1387 w_k1_IBM_s102
                                 + 1.2147 w_k1_INTC_s102 + 1.09988 w_k1_KO_s102
                                 + 1.13842 w_k1_MAT_s102
                                 + 1.24541 w_k1_MSFT_s102
                                 + 0.972095 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.974949 w_k1_ADM_s102
                                 + 0.969683 w_k1_AXP_s102
                                 + 1.00052 w_k1_DIS_s102
                                 + 1.07565 w_k1_FDX_s102
                                 + 0.899078 w_k1_IBM_s102
                                 + 1.11483 w_k1_INTC_s102
                                 + 0.882532 w_k1_KO_s102
                                 + 1.03253 w_k1_MAT_s102
                                 + 1.01628 w_k1_MSFT_s102
                                 + 0.970913 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.882129 w_k1_ADM_s103
                                 + 0.811304 w_k1_AXP_s103
                                 + 0.717605 w_k1_DIS_s103
                                 + 0.78034 w_k1_FDX_s103
                                 + 0.73155 w_k1_IBM_s103
                                 + 0.804833 w_k1_INTC_s103
                                 + 0.848985 w_k1_KO_s103 + 0.6433 w_k1_MAT_s103
                                 + 0.984538 w_k1_MSFT_s103
                                 + 0.848196 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.28193 w_k1_ADM_s103
                                 + 1.36196 w_k1_AXP_s103
                                 + 1.39161 w_k1_DIS_s103
                                 + 1.15894 w_k1_FDX_s103
                                 + 1.61376 w_k1_IBM_s103
                                 + 1.17496 w_k1_INTC_s103
                                 + 1.50492 w_k1_KO_s103 + 1.09162 w_k1_MAT_s103
                                 + 0.998017 w_k1_MSFT_s103
                                 + 1.0623 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.967155 w_k1_ADM_s103
                                 + 0.989314 w_k1_AXP_s103
                                 + 0.85087 w_k1_DIS_s103
                                 + 0.807512 w_k1_FDX_s103
                                 + 1.11369 w_k1_IBM_s103
                                 + 0.901462 w_k1_INTC_s103
                                 + 1.1151 w_k1_KO_s103 + 1.07625 w_k1_MAT_s103
                                 + 0.956028 w_k1_MSFT_s103
                                 + 1.06729 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.25546 w_k1_ADM_s103
                                 + 0.979211 w_k1_AXP_s103
                                 + 1.25394 w_k1_DIS_s103
                                 + 1.09687 w_k1_FDX_s103
                                 + 0.841493 w_k1_IBM_s103
                                 + 1.00976 w_k1_INTC_s103
                                 + 1.13201 w_k1_KO_s103
                                 + 0.866706 w_k1_MAT_s103
                                 + 1.11453 w_k1_MSFT_s103
                                 + 1.11835 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.01685 w_k1_ADM_s103
                                 + 1.11011 w_k1_AXP_s103
                                 + 1.11191 w_k1_DIS_s103
                                 + 0.90511 w_k1_FDX_s103
                                 + 1.24543 w_k1_IBM_s103
                                 + 0.919177 w_k1_INTC_s103
                                 + 1.26117 w_k1_KO_s103 + 1.10342 w_k1_MAT_s103
                                 + 0.870708 w_k1_MSFT_s103
                                 + 0.963064 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.3916 w_k1_ADM_s103
                                 + 1.19046 w_k1_AXP_s103
                                 + 1.58305 w_k1_DIS_s103 + 1.3903 w_k1_FDX_s103
                                 + 1.07512 w_k1_IBM_s103
                                 + 1.32332 w_k1_INTC_s103
                                 + 1.32184 w_k1_KO_s103
                                 + 0.976986 w_k1_MAT_s103
                                 + 1.28683 w_k1_MSFT_s103
                                 + 1.09079 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.22575 w_k1_ADM_s103
                                 + 1.20514 w_k1_AXP_s103
                                 + 1.08217 w_k1_DIS_s103
                                 + 1.04331 w_k1_FDX_s103
                                 + 0.836375 w_k1_IBM_s103
                                 + 1.02624 w_k1_INTC_s103
                                 + 1.06098 w_k1_KO_s103
                                 + 0.694303 w_k1_MAT_s103
                                 + 0.92851 w_k1_MSFT_s103
                                 + 0.914143 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.903472 w_k1_ADM_s103
                                 + 0.863996 w_k1_AXP_s103
                                 + 0.677953 w_k1_DIS_s103
                                 + 0.82639 w_k1_FDX_s103
                                 + 0.881277 w_k1_IBM_s103
                                 + 0.92066 w_k1_INTC_s103
                                 + 1.03791 w_k1_KO_s103
                                 + 0.733092 w_k1_MAT_s103
                                 + 0.853899 w_k1_MSFT_s103
                                 + 0.870524 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.0198 w_k1_ADM_s104
                                 + 1.27325 w_k1_AXP_s104
                                 + 1.50286 w_k1_DIS_s104
                                 + 1.18594 w_k1_FDX_s104 + 1.3733 w_k1_IBM_s104
                                 + 1.56378 w_k1_INTC_s104
                                 + 1.32257 w_k1_KO_s104
                                 + 0.959013 w_k1_MAT_s104
                                 + 1.13967 w_k1_MSFT_s104
                                 + 1.10988 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.07894 w_k1_ADM_s104
                                 + 0.950586 w_k1_AXP_s104
                                 + 0.936689 w_k1_DIS_s104
                                 + 0.957271 w_k1_FDX_s104
                                 + 0.852326 w_k1_IBM_s104
                                 + 0.882489 w_k1_INTC_s104
                                 + 0.9423 w_k1_KO_s104 + 0.896636 w_k1_MAT_s104
                                 + 0.938154 w_k1_MSFT_s104
                                 + 1.01639 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.54476 w_k1_ADM_s104
                                 + 1.64563 w_k1_AXP_s104
                                 + 1.16589 w_k1_DIS_s104
                                 + 1.21608 w_k1_FDX_s104
                                 + 0.939506 w_k1_IBM_s104
                                 + 1.06007 w_k1_INTC_s104
                                 + 0.868946 w_k1_KO_s104
                                 + 0.929986 w_k1_MAT_s104
                                 + 0.987206 w_k1_MSFT_s104
                                 + 1.21239 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.71843 w_k1_ADM_s104
                                 + 1.35565 w_k1_AXP_s104
                                 + 1.78712 w_k1_DIS_s104
                                 + 1.41554 w_k1_FDX_s104
                                 + 0.927403 w_k1_IBM_s104
                                 + 0.949828 w_k1_INTC_s104
                                 + 1.15727 w_k1_KO_s104 + 0.79102 w_k1_MAT_s104
                                 + 1.25657 w_k1_MSFT_s104
                                 + 1.25677 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.792655 w_k1_ADM_s104
                                 + 0.756087 w_k1_AXP_s104
                                 + 0.806376 w_k1_DIS_s104
                                 + 0.766312 w_k1_FDX_s104
                                 + 1.03615 w_k1_IBM_s104
                                 + 0.763851 w_k1_INTC_s104
                                 + 0.83858 w_k1_KO_s104
                                 + 0.692108 w_k1_MAT_s104
                                 + 0.721143 w_k1_MSFT_s104
                                 + 0.906523 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.14362 w_k1_ADM_s104
                                 + 1.06414 w_k1_AXP_s104
                                 + 0.89175 w_k1_DIS_s104
                                 + 0.852884 w_k1_FDX_s104
                                 + 1.53042 w_k1_IBM_s104
                                 + 1.04906 w_k1_INTC_s104
                                 + 1.26585 w_k1_KO_s104 + 1.50775 w_k1_MAT_s104
                                 + 0.904235 w_k1_MSFT_s104
                                 + 1.18166 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.25109 w_k1_ADM_s104
                                 + 1.0404 w_k1_AXP_s104 + 1.41979 w_k1_DIS_s104
                                 + 1.12704 w_k1_FDX_s104
                                 + 1.13735 w_k1_IBM_s104
                                 + 1.09705 w_k1_INTC_s104
                                 + 1.02576 w_k1_KO_s104
                                 + 0.749337 w_k1_MAT_s104
                                 + 0.927706 w_k1_MSFT_s104
                                 + 1.15008 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.879874 w_k1_ADM_s104
                                 + 0.780776 w_k1_AXP_s104
                                 + 1.00483 w_k1_DIS_s104
                                 + 0.88996 w_k1_FDX_s104
                                 + 0.674027 w_k1_IBM_s104
                                 + 0.885274 w_k1_INTC_s104
                                 + 0.813128 w_k1_KO_s104
                                 + 0.563877 w_k1_MAT_s104
                                 + 0.833707 w_k1_MSFT_s104
                                 + 0.909189 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.938762 w_k1_ADM_s105
                                 + 0.884342 w_k1_AXP_s105
                                 + 0.805814 w_k1_DIS_s105
                                 + 0.934495 w_k1_FDX_s105
                                 + 0.843472 w_k1_IBM_s105
                                 + 0.92994 w_k1_INTC_s105
                                 + 0.781663 w_k1_KO_s105
                                 + 0.684441 w_k1_MAT_s105
                                 + 0.822689 w_k1_MSFT_s105
                                 + 0.865727 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.08497 w_k1_ADM_s105
                                 + 0.969914 w_k1_AXP_s105
                                 + 0.98113 w_k1_DIS_s105
                                 + 0.881186 w_k1_FDX_s105
                                 + 1.46344 w_k1_IBM_s105
                                 + 0.996765 w_k1_INTC_s105
                                 + 1.03159 w_k1_KO_s105 + 1.35322 w_k1_MAT_s105
                                 + 0.939303 w_k1_MSFT_s105
                                 + 0.954236 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.039 w_k1_ADM_s105
                                 + 0.849453 w_k1_AXP_s105
                                 + 1.36382 w_k1_DIS_s105 + 1.1667 w_k1_FDX_s105
                                 + 0.836327 w_k1_IBM_s105
                                 + 1.23776 w_k1_INTC_s105
                                 + 1.12098 w_k1_KO_s105
                                 + 0.859426 w_k1_MAT_s105
                                 + 1.28631 w_k1_MSFT_s105
                                 + 1.00831 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.35086 w_k1_ADM_s105
                                 + 1.13299 w_k1_AXP_s105
                                 + 1.42406 w_k1_DIS_s105
                                 + 1.23537 w_k1_FDX_s105
                                 + 1.18103 w_k1_IBM_s105
                                 + 1.05481 w_k1_INTC_s105
                                 + 1.31733 w_k1_KO_s105 + 1.51009 w_k1_MAT_s105
                                 + 1.18637 w_k1_MSFT_s105
                                 + 1.23456 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.26178 w_k1_ADM_s105
                                 + 1.01283 w_k1_AXP_s105
                                 + 0.904414 w_k1_DIS_s105
                                 + 1.18519 w_k1_FDX_s105
                                 + 0.712903 w_k1_IBM_s105
                                 + 0.845354 w_k1_INTC_s105
                                 + 0.940072 w_k1_KO_s105
                                 + 0.880093 w_k1_MAT_s105
                                 + 0.936001 w_k1_MSFT_s105
                                 + 0.961564 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.739628 w_k1_ADM_s105
                                 + 0.671456 w_k1_AXP_s105
                                 + 0.714976 w_k1_DIS_s105
                                 + 0.88152 w_k1_FDX_s105
                                 + 0.636505 w_k1_IBM_s105
                                 + 1.00792 w_k1_INTC_s105
                                 + 0.67878 w_k1_KO_s105
                                 + 0.497036 w_k1_MAT_s105
                                 + 0.932231 w_k1_MSFT_s105
                                 + 0.795706 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.939312 w_k1_ADM_s105
                                 + 0.770758 w_k1_AXP_s105
                                 + 0.792622 w_k1_DIS_s105
                                 + 0.852426 w_k1_FDX_s105
                                 + 0.915561 w_k1_IBM_s105
                                 + 1.02733 w_k1_INTC_s105
                                 + 0.847082 w_k1_KO_s105
                                 + 0.723422 w_k1_MAT_s105
                                 + 0.887417 w_k1_MSFT_s105
                                 + 0.901005 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.919116 w_k1_ADM_s105
                                 + 1.1677 w_k1_AXP_s105 + 1.16673 w_k1_DIS_s105
                                 + 1.22045 w_k1_FDX_s105
                                 + 0.985232 w_k1_IBM_s105
                                 + 1.10834 w_k1_INTC_s105 + 1.0611 w_k1_KO_s105
                                 + 0.882168 w_k1_MAT_s105
                                 + 1.00495 w_k1_MSFT_s105
                                 + 0.986864 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.651376 w_k1_ADM_s106
                                 + 0.650644 w_k1_AXP_s106
                                 + 0.616851 w_k1_DIS_s106
                                 + 0.792141 w_k1_FDX_s106
                                 + 0.692878 w_k1_IBM_s106
                                 + 0.93531 w_k1_INTC_s106
                                 + 0.756619 w_k1_KO_s106
                                 + 0.405073 w_k1_MAT_s106
                                 + 0.812907 w_k1_MSFT_s106
                                 + 0.860753 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.31134 w_k1_ADM_s106
                                 + 0.767629 w_k1_AXP_s106
                                 + 0.972798 w_k1_DIS_s106
                                 + 0.972627 w_k1_FDX_s106
                                 + 0.880573 w_k1_IBM_s106
                                 + 0.81043 w_k1_INTC_s106
                                 + 1.00242 w_k1_KO_s106
                                 + 0.689085 w_k1_MAT_s106
                                 + 0.904286 w_k1_MSFT_s106
                                 + 1.1262 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.938222 w_k1_ADM_s106
                                 + 0.829483 w_k1_AXP_s106
                                 + 0.649658 w_k1_DIS_s106
                                 + 0.758825 w_k1_FDX_s106
                                 + 1.02777 w_k1_IBM_s106
                                 + 0.779809 w_k1_INTC_s106
                                 + 0.980763 w_k1_KO_s106
                                 + 1.62775 w_k1_MAT_s106
                                 + 0.788252 w_k1_MSFT_s106
                                 + 0.944343 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.53121 w_k1_ADM_s106
                                 + 1.38254 w_k1_AXP_s106
                                 + 1.96955 w_k1_DIS_s106
                                 + 1.44272 w_k1_FDX_s106
                                 + 1.58342 w_k1_IBM_s106
                                 + 1.44442 w_k1_INTC_s106
                                 + 1.82734 w_k1_KO_s106 + 1.3709 w_k1_MAT_s106
                                 + 1.39033 w_k1_MSFT_s106
                                 + 1.52864 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.961659 w_k1_ADM_s106
                                 + 1.01943 w_k1_AXP_s106
                                 + 0.752205 w_k1_DIS_s106
                                 + 0.96039 w_k1_FDX_s106
                                 + 0.658561 w_k1_IBM_s106
                                 + 0.953934 w_k1_INTC_s106
                                 + 0.819023 w_k1_KO_s106
                                 + 0.953831 w_k1_MAT_s106
                                 + 0.929768 w_k1_MSFT_s106
                                 + 0.982061 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.03921 w_k1_ADM_s106
                                 + 0.976849 w_k1_AXP_s106
                                 + 1.13053 w_k1_DIS_s106
                                 + 1.10703 w_k1_FDX_s106
                                 + 0.825909 w_k1_IBM_s106
                                 + 1.06582 w_k1_INTC_s106
                                 + 1.05253 w_k1_KO_s106
                                 + 0.977925 w_k1_MAT_s106
                                 + 1.00919 w_k1_MSFT_s106
                                 + 1.08645 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.08225 w_k1_ADM_s106
                                 + 1.19828 w_k1_AXP_s106
                                 + 1.15578 w_k1_DIS_s106
                                 + 0.905849 w_k1_FDX_s106
                                 + 1.44758 w_k1_IBM_s106
                                 + 0.835793 w_k1_INTC_s106
                                 + 1.20457 w_k1_KO_s106 + 1.2582 w_k1_MAT_s106
                                 + 0.860219 w_k1_MSFT_s106
                                 + 1.1726 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.12865 w_k1_ADM_s106
                                 + 1.32673 w_k1_AXP_s106
                                 + 1.17898 w_k1_DIS_s106
                                 + 1.17869 w_k1_FDX_s106
                                 + 1.20096 w_k1_IBM_s106
                                 + 0.97866 w_k1_INTC_s106 + 1.1202 w_k1_KO_s106
                                 + 1.46397 w_k1_MAT_s106
                                 + 0.987504 w_k1_MSFT_s106
                                 + 1.13817 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.03084 w_k1_ADM_s107
                                 + 1.05063 w_k1_AXP_s107
                                 + 0.934335 w_k1_DIS_s107
                                 + 1.03333 w_k1_FDX_s107
                                 + 1.06072 w_k1_IBM_s107
                                 + 1.04449 w_k1_INTC_s107
                                 + 0.955889 w_k1_KO_s107
                                 + 0.911388 w_k1_MAT_s107
                                 + 0.916218 w_k1_MSFT_s107
                                 + 1.05545 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.19666 w_k1_ADM_s107
                                 + 1.46457 w_k1_AXP_s107
                                 + 1.70938 w_k1_DIS_s107
                                 + 1.36186 w_k1_FDX_s107
                                 + 1.58342 w_k1_IBM_s107
                                 + 1.36878 w_k1_INTC_s107 + 1.5097 w_k1_KO_s107
                                 + 1.74924 w_k1_MAT_s107
                                 + 1.31558 w_k1_MSFT_s107
                                 + 1.21723 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.10433 w_k1_ADM_s107
                                 + 1.07846 w_k1_AXP_s107
                                 + 1.12646 w_k1_DIS_s107
                                 + 1.02644 w_k1_FDX_s107
                                 + 1.54849 w_k1_IBM_s107
                                 + 1.14422 w_k1_INTC_s107
                                 + 1.47354 w_k1_KO_s107 + 1.55346 w_k1_MAT_s107
                                 + 1.10038 w_k1_MSFT_s107
                                 + 1.19376 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.70668 w_k1_ADM_s107
                                 + 0.887799 w_k1_AXP_s107
                                 + 0.624296 w_k1_DIS_s107
                                 + 0.792539 w_k1_FDX_s107
                                 + 1.1864 w_k1_IBM_s107
                                 + 0.93367 w_k1_INTC_s107
                                 + 1.06633 w_k1_KO_s107 + 1.70738 w_k1_MAT_s107
                                 + 0.862413 w_k1_MSFT_s107
                                 + 0.939409 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.883598 w_k1_ADM_s107
                                 + 0.880993 w_k1_AXP_s107
                                 + 0.780705 w_k1_DIS_s107
                                 + 0.998228 w_k1_FDX_s107
                                 + 0.695356 w_k1_IBM_s107
                                 + 0.909888 w_k1_INTC_s107
                                 + 0.777204 w_k1_KO_s107
                                 + 0.772235 w_k1_MAT_s107
                                 + 0.857326 w_k1_MSFT_s107
                                 + 0.857488 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.977285 w_k1_ADM_s107
                                 + 0.780838 w_k1_AXP_s107
                                 + 0.911407 w_k1_DIS_s107
                                 + 0.945696 w_k1_FDX_s107
                                 + 1.03536 w_k1_IBM_s107
                                 + 1.03531 w_k1_INTC_s107 + 1.0328 w_k1_KO_s107
                                 + 1.05515 w_k1_MAT_s107
                                 + 1.07072 w_k1_MSFT_s107
                                 + 1.07541 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.944193 w_k1_ADM_s107
                                 + 0.701514 w_k1_AXP_s107
                                 + 0.575734 w_k1_DIS_s107
                                 + 0.730574 w_k1_FDX_s107
                                 + 0.855729 w_k1_IBM_s107
                                 + 0.699206 w_k1_INTC_s107
                                 + 0.719428 w_k1_KO_s107
                                 + 0.948265 w_k1_MAT_s107
                                 + 0.683892 w_k1_MSFT_s107
                                 + 0.907862 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.07206 w_k1_ADM_s107
                                 + 1.54441 w_k1_AXP_s107
                                 + 1.09743 w_k1_DIS_s107
                                 + 1.29435 w_k1_FDX_s107
                                 + 0.927494 w_k1_IBM_s107
                                 + 1.28387 w_k1_INTC_s107
                                 + 1.01921 w_k1_KO_s107 + 1.31699 w_k1_MAT_s107
                                 + 1.05213 w_k1_MSFT_s107
                                 + 1.17854 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.60511 w_k1_ADM_s110
                                 + 0.734841 w_k1_AXP_s110
                                 + 0.603423 w_k1_DIS_s110
                                 + 0.791647 w_k1_FDX_s110
                                 + 0.681251 w_k1_IBM_s110
                                 + 0.912542 w_k1_INTC_s110
                                 + 0.65293 w_k1_KO_s110
                                 + 0.624238 w_k1_MAT_s110
                                 + 0.7223 w_k1_MSFT_s110
                                 + 0.850874 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01467 w_k1_ADM_s110
                                 + 1.09339 w_k1_AXP_s110 + 1.2691 w_k1_DIS_s110
                                 + 1.05142 w_k1_FDX_s110
                                 + 1.17278 w_k1_IBM_s110
                                 + 1.00809 w_k1_INTC_s110
                                 + 1.08402 w_k1_KO_s110
                                 + 0.975627 w_k1_MAT_s110
                                 + 1.03259 w_k1_MSFT_s110
                                 + 1.08688 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.30597 w_k1_ADM_s110
                                 + 1.27823 w_k1_AXP_s110
                                 + 1.60715 w_k1_DIS_s110
                                 + 1.24089 w_k1_FDX_s110
                                 + 1.49845 w_k1_IBM_s110
                                 + 1.14439 w_k1_INTC_s110
                                 + 1.43071 w_k1_KO_s110 + 1.47412 w_k1_MAT_s110
                                 + 1.17452 w_k1_MSFT_s110
                                 + 1.25154 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.26815 w_k1_ADM_s110
                                 + 1.38522 w_k1_AXP_s110
                                 + 1.42199 w_k1_DIS_s110
                                 + 1.32307 w_k1_FDX_s110
                                 + 1.00467 w_k1_IBM_s110
                                 + 1.12808 w_k1_INTC_s110 + 1.1434 w_k1_KO_s110
                                 + 1.08863 w_k1_MAT_s110
                                 + 1.20535 w_k1_MSFT_s110
                                 + 1.13719 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.924952 w_k1_ADM_s110
                                 + 0.897138 w_k1_AXP_s110
                                 + 0.95445 w_k1_DIS_s110
                                 + 1.01183 w_k1_FDX_s110
                                 + 0.804892 w_k1_IBM_s110
                                 + 0.941521 w_k1_INTC_s110
                                 + 0.897795 w_k1_KO_s110
                                 + 0.574907 w_k1_MAT_s110
                                 + 0.904598 w_k1_MSFT_s110
                                 + 0.942647 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.804786 w_k1_ADM_s110
                                 + 0.784412 w_k1_AXP_s110
                                 + 0.727671 w_k1_DIS_s110
                                 + 0.808902 w_k1_FDX_s110
                                 + 1.00836 w_k1_IBM_s110
                                 + 0.849785 w_k1_INTC_s110
                                 + 0.919011 w_k1_KO_s110
                                 + 0.847894 w_k1_MAT_s110
                                 + 0.870055 w_k1_MSFT_s110
                                 + 0.945311 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.00429 w_k1_ADM_s110
                                 + 0.950364 w_k1_AXP_s110
                                 + 0.786465 w_k1_DIS_s110
                                 + 0.846408 w_k1_FDX_s110
                                 + 0.837711 w_k1_IBM_s110
                                 + 0.921528 w_k1_INTC_s110
                                 + 0.860006 w_k1_KO_s110
                                 + 0.794193 w_k1_MAT_s110
                                 + 0.865378 w_k1_MSFT_s110
                                 + 1.01999 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.92153 w_k1_ADM_s110
                                 + 1.03593 w_k1_AXP_s110
                                 + 1.04244 w_k1_DIS_s110
                                 + 1.01178 w_k1_FDX_s110
                                 + 1.28094 w_k1_IBM_s110
                                 + 0.955902 w_k1_INTC_s110
                                 + 1.09676 w_k1_KO_s110 + 1.54164 w_k1_MAT_s110
                                 + 0.910986 w_k1_MSFT_s110 + 1.076 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.37366 w_k1_ADM_s111
                                 + 1.34661 w_k1_AXP_s111
                                 + 1.65688 w_k1_DIS_s111
                                 + 1.32812 w_k1_FDX_s111 + 1.2068 w_k1_IBM_s111
                                 + 1.40707 w_k1_INTC_s111
                                 + 1.32532 w_k1_KO_s111 + 1.33085 w_k1_MAT_s111
                                 + 1.31305 w_k1_MSFT_s111
                                 + 1.04961 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.04335 w_k1_ADM_s111
                                 + 0.96599 w_k1_AXP_s111
                                 + 0.973063 w_k1_DIS_s111
                                 + 1.06665 w_k1_FDX_s111
                                 + 0.849901 w_k1_IBM_s111
                                 + 0.984287 w_k1_INTC_s111
                                 + 0.942822 w_k1_KO_s111
                                 + 1.13368 w_k1_MAT_s111
                                 + 1.03812 w_k1_MSFT_s111 + 1.0159 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.902215 w_k1_ADM_s111
                                 + 0.623802 w_k1_AXP_s111
                                 + 0.636843 w_k1_DIS_s111
                                 + 0.817434 w_k1_FDX_s111
                                 + 0.56168 w_k1_IBM_s111
                                 + 0.820147 w_k1_INTC_s111
                                 + 0.69157 w_k1_KO_s111
                                 + 0.727591 w_k1_MAT_s111
                                 + 0.815425 w_k1_MSFT_s111
                                 + 0.805938 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.813714 w_k1_ADM_s111
                                 + 0.769544 w_k1_AXP_s111
                                 + 0.743636 w_k1_DIS_s111
                                 + 0.814232 w_k1_FDX_s111
                                 + 0.786638 w_k1_IBM_s111
                                 + 1.03771 w_k1_INTC_s111
                                 + 0.859927 w_k1_KO_s111
                                 + 0.78974 w_k1_MAT_s111
                                 + 0.92751 w_k1_MSFT_s111 + 0.9167 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.963476 w_k1_ADM_s111
                                 + 1.33748 w_k1_AXP_s111
                                 + 0.971819 w_k1_DIS_s111
                                 + 1.01634 w_k1_FDX_s111
                                 + 1.26245 w_k1_IBM_s111
                                 + 0.962729 w_k1_INTC_s111
                                 + 1.18567 w_k1_KO_s111 + 1.74788 w_k1_MAT_s111
                                 + 0.950743 w_k1_MSFT_s111
                                 + 1.08242 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.792569 w_k1_ADM_s111
                                 + 0.832709 w_k1_AXP_s111
                                 + 0.856793 w_k1_DIS_s111
                                 + 0.845104 w_k1_FDX_s111
                                 + 0.996395 w_k1_IBM_s111
                                 + 1.04607 w_k1_INTC_s111
                                 + 0.934179 w_k1_KO_s111
                                 + 0.98147 w_k1_MAT_s111
                                 + 0.841941 w_k1_MSFT_s111
                                 + 0.991367 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.10531 w_k1_ADM_s111
                                 + 1.49874 w_k1_AXP_s111
                                 + 1.53936 w_k1_DIS_s111
                                 + 1.13842 w_k1_FDX_s111
                                 + 1.86067 w_k1_IBM_s111
                                 + 1.42495 w_k1_INTC_s111
                                 + 1.77133 w_k1_KO_s111 + 1.9627 w_k1_MAT_s111
                                 + 1.25036 w_k1_MSFT_s111
                                 + 1.19374 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.889063 w_k1_ADM_s111
                                 + 0.941417 w_k1_AXP_s111
                                 + 1.23333 w_k1_DIS_s111
                                 + 0.999748 w_k1_FDX_s111
                                 + 1.15317 w_k1_IBM_s111
                                 + 1.17707 w_k1_INTC_s111 + 1.3056 w_k1_KO_s111
                                 + 0.73744 w_k1_MAT_s111
                                 + 1.09349 w_k1_MSFT_s111
                                 + 0.971953 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.19643 w_k1_ADM_s112
                                 + 1.35532 w_k1_AXP_s112
                                 + 1.68673 w_k1_DIS_s112
                                 + 1.34351 w_k1_FDX_s112
                                 + 1.38932 w_k1_IBM_s112
                                 + 1.6491 w_k1_INTC_s112 + 1.38921 w_k1_KO_s112
                                 + 1.34485 w_k1_MAT_s112
                                 + 1.32167 w_k1_MSFT_s112
                                 + 1.17639 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.00608 w_k1_ADM_s112
                                 + 1.17349 w_k1_AXP_s112
                                 + 1.10368 w_k1_DIS_s112
                                 + 1.13637 w_k1_FDX_s112
                                 + 1.39857 w_k1_IBM_s112
                                 + 1.33831 w_k1_INTC_s112
                                 + 1.28897 w_k1_KO_s112 + 1.52367 w_k1_MAT_s112
                                 + 1.0852 w_k1_MSFT_s112 + 1.08859 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.909795 w_k1_ADM_s112
                                 + 1.02911 w_k1_AXP_s112
                                 + 0.860589 w_k1_DIS_s112
                                 + 0.889961 w_k1_FDX_s112
                                 + 0.992023 w_k1_IBM_s112
                                 + 1.16932 w_k1_INTC_s112
                                 + 0.98163 w_k1_KO_s112 + 1.18348 w_k1_MAT_s112
                                 + 0.87201 w_k1_MSFT_s112
                                 + 0.946249 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06202 w_k1_ADM_s112
                                 + 1.32829 w_k1_AXP_s112
                                 + 1.00769 w_k1_DIS_s112
                                 + 1.17877 w_k1_FDX_s112
                                 + 0.777388 w_k1_IBM_s112
                                 + 0.979227 w_k1_INTC_s112
                                 + 0.9113 w_k1_KO_s112 + 1.00296 w_k1_MAT_s112
                                 + 1.04102 w_k1_MSFT_s112
                                 + 0.950512 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.07303 w_k1_ADM_s112
                                 + 1.05891 w_k1_AXP_s112
                                 + 0.608886 w_k1_DIS_s112
                                 + 0.910947 w_k1_FDX_s112
                                 + 0.675885 w_k1_IBM_s112
                                 + 0.937816 w_k1_INTC_s112
                                 + 0.677001 w_k1_KO_s112
                                 + 0.859857 w_k1_MAT_s112
                                 + 0.860453 w_k1_MSFT_s112
                                 + 0.851132 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.873983 w_k1_ADM_s112
                                 + 0.780624 w_k1_AXP_s112
                                 + 0.77486 w_k1_DIS_s112
                                 + 0.833855 w_k1_FDX_s112
                                 + 0.861541 w_k1_IBM_s112
                                 + 1.03106 w_k1_INTC_s112
                                 + 0.958673 w_k1_KO_s112
                                 + 0.96493 w_k1_MAT_s112
                                 + 0.928303 w_k1_MSFT_s112
                                 + 0.910773 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.28423 w_k1_ADM_s112
                                 + 1.16932 w_k1_AXP_s112
                                 + 1.13865 w_k1_DIS_s112
                                 + 1.04749 w_k1_FDX_s112
                                 + 1.05467 w_k1_IBM_s112
                                 + 1.19083 w_k1_INTC_s112
                                 + 1.11126 w_k1_KO_s112 + 1.54122 w_k1_MAT_s112
                                 + 1.07742 w_k1_MSFT_s112
                                 + 0.968327 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.29739 w_k1_ADM_s112
                                 + 1.09674 w_k1_AXP_s112
                                 + 1.48739 w_k1_DIS_s112 + 1.1483 w_k1_FDX_s112
                                 + 0.833842 w_k1_IBM_s112
                                 + 1.30672 w_k1_INTC_s112
                                 + 1.29692 w_k1_KO_s112
                                 + 0.897453 w_k1_MAT_s112
                                 + 1.26954 w_k1_MSFT_s112
                                 + 0.95326 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.922223 w_k1_ADM_s113
                                 + 0.756917 w_k1_AXP_s113
                                 + 0.687647 w_k1_DIS_s113
                                 + 0.914525 w_k1_FDX_s113
                                 + 0.648207 w_k1_IBM_s113
                                 + 0.852184 w_k1_INTC_s113
                                 + 0.676548 w_k1_KO_s113
                                 + 0.494469 w_k1_MAT_s113
                                 + 0.770434 w_k1_MSFT_s113
                                 + 0.78665 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.958228 w_k1_ADM_s113
                                 + 0.796797 w_k1_AXP_s113
                                 + 0.869536 w_k1_DIS_s113
                                 + 1.03763 w_k1_FDX_s113
                                 + 0.741336 w_k1_IBM_s113
                                 + 0.995495 w_k1_INTC_s113
                                 + 0.884929 w_k1_KO_s113
                                 + 0.799643 w_k1_MAT_s113
                                 + 0.959783 w_k1_MSFT_s113
                                 + 0.931094 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.1379 w_k1_ADM_s113
                                 + 1.07967 w_k1_AXP_s113 + 1.3081 w_k1_DIS_s113
                                 + 1.17225 w_k1_FDX_s113
                                 + 1.25355 w_k1_IBM_s113
                                 + 1.03386 w_k1_INTC_s113
                                 + 1.05814 w_k1_KO_s113 + 1.01991 w_k1_MAT_s113
                                 + 0.965753 w_k1_MSFT_s113
                                 + 1.04939 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11797 w_k1_ADM_s113
                                 + 1.19244 w_k1_AXP_s113
                                 + 1.39154 w_k1_DIS_s113 + 1.1749 w_k1_FDX_s113
                                 + 1.76388 w_k1_IBM_s113
                                 + 0.866161 w_k1_INTC_s113
                                 + 1.20056 w_k1_KO_s113 + 1.45123 w_k1_MAT_s113
                                 + 0.839566 w_k1_MSFT_s113
                                 + 1.08457 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.16257 w_k1_ADM_s113
                                 + 0.956313 w_k1_AXP_s113
                                 + 1.63056 w_k1_DIS_s113
                                 + 1.27205 w_k1_FDX_s113
                                 + 0.926159 w_k1_IBM_s113
                                 + 1.14065 w_k1_INTC_s113
                                 + 1.08561 w_k1_KO_s113
                                 + 0.814816 w_k1_MAT_s113
                                 + 1.14017 w_k1_MSFT_s113
                                 + 1.07168 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.808804 w_k1_ADM_s113
                                 + 0.859533 w_k1_AXP_s113
                                 + 0.812062 w_k1_DIS_s113
                                 + 0.859611 w_k1_FDX_s113
                                 + 1.02418 w_k1_IBM_s113
                                 + 0.848021 w_k1_INTC_s113
                                 + 0.889528 w_k1_KO_s113
                                 + 0.843549 w_k1_MAT_s113
                                 + 0.706123 w_k1_MSFT_s113
                                 + 0.791201 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.07707 w_k1_ADM_s113
                                 + 1.04683 w_k1_AXP_s113
                                 + 1.14663 w_k1_DIS_s113
                                 + 1.22917 w_k1_FDX_s113
                                 + 0.71459 w_k1_IBM_s113
                                 + 0.924761 w_k1_INTC_s113
                                 + 0.809468 w_k1_KO_s113
                                 + 0.756184 w_k1_MAT_s113
                                 + 0.947634 w_k1_MSFT_s113
                                 + 0.990503 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.829653 w_k1_ADM_s113
                                 + 0.496631 w_k1_AXP_s113
                                 + 0.492136 w_k1_DIS_s113
                                 + 0.704587 w_k1_FDX_s113
                                 + 0.613075 w_k1_IBM_s113
                                 + 0.727531 w_k1_INTC_s113
                                 + 0.610813 w_k1_KO_s113
                                 + 0.352225 w_k1_MAT_s113
                                 + 0.697546 w_k1_MSFT_s113
                                 + 0.750055 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.73674 w_k1_ADM_s114
                                 + 1.87997 w_k1_AXP_s114
                                 + 2.06115 w_k1_DIS_s114
                                 + 1.46897 w_k1_FDX_s114
                                 + 1.26639 w_k1_IBM_s114
                                 + 1.06243 w_k1_INTC_s114
                                 + 1.48602 w_k1_KO_s114 + 1.64036 w_k1_MAT_s114
                                 + 1.22465 w_k1_MSFT_s114
                                 + 1.33738 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.12795 w_k1_ADM_s114
                                 + 1.11003 w_k1_AXP_s114
                                 + 0.94786 w_k1_DIS_s114 + 1.0177 w_k1_FDX_s114
                                 + 0.755761 w_k1_IBM_s114
                                 + 0.846395 w_k1_INTC_s114
                                 + 0.855645 w_k1_KO_s114
                                 + 0.964981 w_k1_MAT_s114
                                 + 0.86238 w_k1_MSFT_s114 + 1.0048 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.09594 w_k1_ADM_s114
                                 + 1.2981 w_k1_AXP_s114 + 1.30135 w_k1_DIS_s114
                                 + 1.1695 w_k1_FDX_s114 + 1.06072 w_k1_IBM_s114
                                 + 1.03474 w_k1_INTC_s114
                                 + 1.08629 w_k1_KO_s114 + 1.27479 w_k1_MAT_s114
                                 + 1.0336 w_k1_MSFT_s114 + 1.16163 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.3319 w_k1_ADM_s114
                                 + 1.04017 w_k1_AXP_s114 + 1.2984 w_k1_DIS_s114
                                 + 1.22147 w_k1_FDX_s114
                                 + 0.763526 w_k1_IBM_s114
                                 + 1.07718 w_k1_INTC_s114
                                 + 0.842916 w_k1_KO_s114
                                 + 0.679919 w_k1_MAT_s114
                                 + 1.08551 w_k1_MSFT_s114
                                 + 0.912763 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.02771 w_k1_ADM_s114
                                 + 1.07931 w_k1_AXP_s114
                                 + 0.603128 w_k1_DIS_s114
                                 + 0.831668 w_k1_FDX_s114
                                 + 0.771748 w_k1_IBM_s114
                                 + 0.856962 w_k1_INTC_s114
                                 + 0.741343 w_k1_KO_s114 + 1.2601 w_k1_MAT_s114
                                 + 0.788278 w_k1_MSFT_s114
                                 + 0.907681 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.31394 w_k1_ADM_s114
                                 + 1.21981 w_k1_AXP_s114
                                 + 1.57994 w_k1_DIS_s114
                                 + 1.08115 w_k1_FDX_s114 + 1.9201 w_k1_IBM_s114
                                 + 1.12025 w_k1_INTC_s114
                                 + 1.66915 w_k1_KO_s114 + 1.29994 w_k1_MAT_s114
                                 + 1.0598 w_k1_MSFT_s114 + 1.14208 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.34078 w_k1_ADM_s114
                                 + 1.34567 w_k1_AXP_s114
                                 + 1.09005 w_k1_DIS_s114
                                 + 1.03909 w_k1_FDX_s114
                                 + 1.45111 w_k1_IBM_s114
                                 + 1.00106 w_k1_INTC_s114
                                 + 1.19557 w_k1_KO_s114 + 1.35973 w_k1_MAT_s114
                                 + 0.947677 w_k1_MSFT_s114
                                 + 0.994861 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.920087 w_k1_ADM_s114
                                 + 0.750628 w_k1_AXP_s114
                                 + 0.815661 w_k1_DIS_s114
                                 + 0.806599 w_k1_FDX_s114
                                 + 0.960189 w_k1_IBM_s114
                                 + 0.712103 w_k1_INTC_s114
                                 + 0.952806 w_k1_KO_s114
                                 + 1.39128 w_k1_MAT_s114
                                 + 0.807171 w_k1_MSFT_s114
                                 + 0.906903 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.975762 w_k1_ADM_s115
                                 + 1.37135 w_k1_AXP_s115 + 1.3531 w_k1_DIS_s115
                                 + 1.20704 w_k1_FDX_s115
                                 + 1.29379 w_k1_IBM_s115
                                 + 1.15466 w_k1_INTC_s115
                                 + 1.25239 w_k1_KO_s115 + 1.30372 w_k1_MAT_s115
                                 + 1.36905 w_k1_MSFT_s115 + 1.1164 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.758706 w_k1_ADM_s115
                                 + 0.834988 w_k1_AXP_s115
                                 + 0.587309 w_k1_DIS_s115
                                 + 0.717656 w_k1_FDX_s115
                                 + 0.746024 w_k1_IBM_s115
                                 + 0.640685 w_k1_INTC_s115
                                 + 0.632937 w_k1_KO_s115
                                 + 0.677737 w_k1_MAT_s115
                                 + 0.802297 w_k1_MSFT_s115
                                 + 0.873845 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.5416 w_k1_ADM_s115
                                 + 1.72429 w_k1_AXP_s115
                                 + 1.58948 w_k1_DIS_s115
                                 + 1.42736 w_k1_FDX_s115
                                 + 1.17159 w_k1_IBM_s115
                                 + 0.960963 w_k1_INTC_s115
                                 + 1.28178 w_k1_KO_s115 + 1.25522 w_k1_MAT_s115
                                 + 1.28898 w_k1_MSFT_s115
                                 + 1.36011 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.772216 w_k1_ADM_s115
                                 + 0.781392 w_k1_AXP_s115
                                 + 0.811372 w_k1_DIS_s115
                                 + 0.886732 w_k1_FDX_s115
                                 + 0.832429 w_k1_IBM_s115
                                 + 0.976667 w_k1_INTC_s115
                                 + 0.737785 w_k1_KO_s115
                                 + 0.688969 w_k1_MAT_s115
                                 + 0.884449 w_k1_MSFT_s115
                                 + 0.979783 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.07655 w_k1_ADM_s115
                                 + 1.04502 w_k1_AXP_s115 + 1.0509 w_k1_DIS_s115
                                 + 1.07536 w_k1_FDX_s115
                                 + 1.57779 w_k1_IBM_s115
                                 + 0.954437 w_k1_INTC_s115
                                 + 1.14586 w_k1_KO_s115 + 1.65015 w_k1_MAT_s115
                                 + 1.29535 w_k1_MSFT_s115
                                 + 1.19354 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.839497 w_k1_ADM_s115
                                 + 0.98683 w_k1_AXP_s115
                                 + 0.929277 w_k1_DIS_s115
                                 + 0.879466 w_k1_FDX_s115
                                 + 1.09102 w_k1_IBM_s115
                                 + 0.963489 w_k1_INTC_s115
                                 + 0.929971 w_k1_KO_s115
                                 + 0.883891 w_k1_MAT_s115
                                 + 0.991829 w_k1_MSFT_s115
                                 + 1.00381 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.17986 w_k1_ADM_s115
                                 + 0.911003 w_k1_AXP_s115
                                 + 1.19125 w_k1_DIS_s115
                                 + 1.08603 w_k1_FDX_s115
                                 + 1.05849 w_k1_IBM_s115
                                 + 0.989738 w_k1_INTC_s115
                                 + 1.1001 w_k1_KO_s115 + 0.671047 w_k1_MAT_s115
                                 + 1.17242 w_k1_MSFT_s115
                                 + 1.10795 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.07904 w_k1_ADM_s115
                                 + 1.21246 w_k1_AXP_s115 + 1.2253 w_k1_DIS_s115
                                 + 1.20518 w_k1_FDX_s115
                                 + 0.776183 w_k1_IBM_s115
                                 + 1.05253 w_k1_INTC_s115
                                 + 0.926772 w_k1_KO_s115
                                 + 0.840609 w_k1_MAT_s115
                                 + 1.22683 w_k1_MSFT_s115
                                 + 1.11728 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.10707 w_k1_ADM_s116
                                 + 0.966795 w_k1_AXP_s116
                                 + 1.07644 w_k1_DIS_s116
                                 + 1.02987 w_k1_FDX_s116
                                 + 0.888294 w_k1_IBM_s116
                                 + 1.04232 w_k1_INTC_s116
                                 + 0.834289 w_k1_KO_s116
                                 + 0.720495 w_k1_MAT_s116
                                 + 0.980187 w_k1_MSFT_s116
                                 + 1.01603 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.05179 w_k1_ADM_s116
                                 + 1.02248 w_k1_AXP_s116
                                 + 0.919245 w_k1_DIS_s116
                                 + 1.15928 w_k1_FDX_s116
                                 + 0.588067 w_k1_IBM_s116
                                 + 1.04687 w_k1_INTC_s116
                                 + 0.804527 w_k1_KO_s116
                                 + 0.593389 w_k1_MAT_s116
                                 + 0.939353 w_k1_MSFT_s116
                                 + 0.955064 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.944054 w_k1_ADM_s116
                                 + 0.823655 w_k1_AXP_s116
                                 + 0.817177 w_k1_DIS_s116
                                 + 0.875343 w_k1_FDX_s116
                                 + 0.949295 w_k1_IBM_s116
                                 + 1.07535 w_k1_INTC_s116
                                 + 0.819503 w_k1_KO_s116
                                 + 0.570836 w_k1_MAT_s116
                                 + 0.810752 w_k1_MSFT_s116
                                 + 0.943778 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.767088 w_k1_ADM_s116
                                 + 0.59281 w_k1_AXP_s116
                                 + 0.599685 w_k1_DIS_s116
                                 + 0.699631 w_k1_FDX_s116
                                 + 0.667957 w_k1_IBM_s116
                                 + 0.833683 w_k1_INTC_s116
                                 + 0.668046 w_k1_KO_s116
                                 + 0.696433 w_k1_MAT_s116
                                 + 0.766407 w_k1_MSFT_s116
                                 + 0.878991 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.25289 w_k1_ADM_s116
                                 + 1.33883 w_k1_AXP_s116
                                 + 1.24733 w_k1_DIS_s116
                                 + 1.19368 w_k1_FDX_s116
                                 + 1.27105 w_k1_IBM_s116
                                 + 1.09243 w_k1_INTC_s116
                                 + 1.28483 w_k1_KO_s116 + 1.03285 w_k1_MAT_s116
                                 + 1.03088 w_k1_MSFT_s116
                                 + 1.18593 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.06477 w_k1_ADM_s116
                                 + 1.10416 w_k1_AXP_s116
                                 + 1.02112 w_k1_DIS_s116
                                 + 0.965545 w_k1_FDX_s116
                                 + 1.13651 w_k1_IBM_s116
                                 + 0.789361 w_k1_INTC_s116
                                 + 0.89567 w_k1_KO_s116 + 1.26119 w_k1_MAT_s116
                                 + 0.820016 w_k1_MSFT_s116
                                 + 0.969762 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.36294 w_k1_ADM_s116
                                 + 1.61955 w_k1_AXP_s116
                                 + 1.55469 w_k1_DIS_s116 + 1.4604 w_k1_FDX_s116
                                 + 1.18161 w_k1_IBM_s116
                                 + 1.18123 w_k1_INTC_s116
                                 + 1.13184 w_k1_KO_s116 + 1.19717 w_k1_MAT_s116
                                 + 1.10832 w_k1_MSFT_s116
                                 + 1.20528 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.87941 w_k1_ADM_s116
                                 + 0.862696 w_k1_AXP_s116
                                 + 1.01944 w_k1_DIS_s116
                                 + 0.897659 w_k1_FDX_s116
                                 + 1.2852 w_k1_IBM_s116
                                 + 1.03406 w_k1_INTC_s116 + 1.2465 w_k1_KO_s116
                                 + 1.11733 w_k1_MAT_s116
                                 + 0.984889 w_k1_MSFT_s116
                                 + 1.05911 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.35529 w_k1_ADM_s117
                                 + 1.09296 w_k1_AXP_s117
                                 + 1.16911 w_k1_DIS_s117
                                 + 0.862802 w_k1_FDX_s117
                                 + 1.58693 w_k1_IBM_s117
                                 + 0.698935 w_k1_INTC_s117
                                 + 1.16587 w_k1_KO_s117 + 1.30349 w_k1_MAT_s117
                                 + 0.884546 w_k1_MSFT_s117
                                 + 1.11605 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.870593 w_k1_ADM_s117
                                 + 0.806221 w_k1_AXP_s117
                                 + 0.784118 w_k1_DIS_s117
                                 + 0.787579 w_k1_FDX_s117
                                 + 0.88107 w_k1_IBM_s117
                                 + 0.819651 w_k1_INTC_s117
                                 + 0.829495 w_k1_KO_s117
                                 + 0.895563 w_k1_MAT_s117
                                 + 0.87893 w_k1_MSFT_s117
                                 + 0.946708 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.90888 w_k1_ADM_s117
                                 + 0.627263 w_k1_AXP_s117
                                 + 0.725816 w_k1_DIS_s117
                                 + 0.76515 w_k1_FDX_s117
                                 + 0.709209 w_k1_IBM_s117
                                 + 0.764557 w_k1_INTC_s117
                                 + 0.777744 w_k1_KO_s117
                                 + 0.622292 w_k1_MAT_s117
                                 + 0.822047 w_k1_MSFT_s117
                                 + 0.802858 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.980832 w_k1_ADM_s117
                                 + 0.890235 w_k1_AXP_s117
                                 + 0.713342 w_k1_DIS_s117
                                 + 0.915145 w_k1_FDX_s117
                                 + 0.572657 w_k1_IBM_s117
                                 + 0.829426 w_k1_INTC_s117
                                 + 0.658526 w_k1_KO_s117
                                 + 0.461806 w_k1_MAT_s117
                                 + 0.865478 w_k1_MSFT_s117
                                 + 0.918474 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.17525 w_k1_ADM_s117
                                 + 1.30115 w_k1_AXP_s117
                                 + 1.14335 w_k1_DIS_s117
                                 + 1.07551 w_k1_FDX_s117
                                 + 1.08802 w_k1_IBM_s117
                                 + 1.0489 w_k1_INTC_s117 + 1.07294 w_k1_KO_s117
                                 + 0.990026 w_k1_MAT_s117
                                 + 1.07669 w_k1_MSFT_s117
                                 + 0.98592 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.4885 w_k1_ADM_s117
                                 + 1.24141 w_k1_AXP_s117
                                 + 1.36048 w_k1_DIS_s117
                                 + 1.26775 w_k1_FDX_s117
                                 + 1.33017 w_k1_IBM_s117
                                 + 0.916578 w_k1_INTC_s117
                                 + 1.25519 w_k1_KO_s117 + 1.34686 w_k1_MAT_s117
                                 + 1.17143 w_k1_MSFT_s117
                                 + 1.23244 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.19419 w_k1_ADM_s117
                                 + 1.15932 w_k1_AXP_s117
                                 + 1.37676 w_k1_DIS_s117 + 1.1198 w_k1_FDX_s117
                                 + 0.972733 w_k1_IBM_s117
                                 + 0.995712 w_k1_INTC_s117
                                 + 1.07789 w_k1_KO_s117 + 1.19918 w_k1_MAT_s117
                                 + 1.1442 w_k1_MSFT_s117 + 1.10111 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.17701 w_k1_ADM_s117
                                 + 0.916637 w_k1_AXP_s117
                                 + 1.25125 w_k1_DIS_s117
                                 + 1.11015 w_k1_FDX_s117
                                 + 0.727104 w_k1_IBM_s117
                                 + 0.910061 w_k1_INTC_s117
                                 + 0.83193 w_k1_KO_s117
                                 + 0.852436 w_k1_MAT_s117
                                 + 1.17187 w_k1_MSFT_s117
                                 + 0.986173 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0122496608761559 w_k1_ADM_s000
                                 + 0.0480314972756415 w_k1_ADM_s001
                                 + 0.036786523591102 w_k1_ADM_s002
                                 + 0.042314921201525 w_k1_ADM_s003
                                 + 0.0373828386887513 w_k1_ADM_s004
                                 + 0.0430907891778427 w_k1_ADM_s005
                                 + 0.0391852437113357 w_k1_ADM_s006
                                 + 0.0539181357253172 w_k1_ADM_s007
                                 + 0.0237539798097788 w_k1_ADM_s010
                                 + 0.0216461879643479 w_k1_ADM_s011
                                 + 0.0144354899968247 w_k1_ADM_s012
                                 + 0.0194480222533961 w_k1_ADM_s013
                                 + 0.0368404952969073 w_k1_ADM_s014
                                 + 0.0242129735060157 w_k1_ADM_s015
                                 + 0.0485420755512025 w_k1_ADM_s016
                                 + 0.0302152690611788 w_k1_ADM_s017
                                 + 0.0571443189208547 w_k1_ADM_s100
                                 + 0.0140940338742776 w_k1_ADM_s101
                                 + 0.0187482501182565 w_k1_ADM_s102
                                 + 0.0126349276853382 w_k1_ADM_s103
                                 + 0.0133322864999729 w_k1_ADM_s104
                                 + 0.0161811467561385 w_k1_ADM_s105
                                 + 0.0124219043392879 w_k1_ADM_s106
                                 + 0.0120816729190536 w_k1_ADM_s107
                                 + 0.0229099358295476 w_k1_ADM_s110
                                 + 0.0815655036826998 w_k1_ADM_s111
                                 + 0.0195322435057374 w_k1_ADM_s112
                                 + 0.0286016461689791 w_k1_ADM_s113
                                 + 0.0494025801332413 w_k1_ADM_s114
                                 + 0.020275138006266 w_k1_ADM_s115
                                 + 0.0313986483442841 w_k1_ADM_s116
                                 + 0.0905079710363808 w_k1_ADM_s117
                                 + 0.0129772298880519 w_k1_AXP_s000
                                 + 0.053243563927456 w_k1_AXP_s001
                                 + 0.0349747997336857 w_k1_AXP_s002
                                 + 0.0484046040314412 w_k1_AXP_s003
                                 + 0.0368347494660119 w_k1_AXP_s004
                                 + 0.0422939267915618 w_k1_AXP_s005
                                 + 0.0433675505970621 w_k1_AXP_s006
                                 + 0.0544500681147697 w_k1_AXP_s007
                                 + 0.020760639547554 w_k1_AXP_s010
                                 + 0.023466712425642 w_k1_AXP_s011
                                 + 0.0152640058584856 w_k1_AXP_s012
                                 + 0.0179815883843407 w_k1_AXP_s013
                                 + 0.0341168993243512 w_k1_AXP_s014
                                 + 0.0214954736111082 w_k1_AXP_s015
                                 + 0.0419953035112478 w_k1_AXP_s016
                                 + 0.0358674631641267 w_k1_AXP_s017
                                 + 0.0533798506691516 w_k1_AXP_s100
                                 + 0.0137293723967945 w_k1_AXP_s101
                                 + 0.0208223726010394 w_k1_AXP_s102
                                 + 0.0120492376706345 w_k1_AXP_s103
                                 + 0.0124427321741665 w_k1_AXP_s104
                                 + 0.0147421269390756 w_k1_AXP_s105
                                 + 0.0121577510706768 w_k1_AXP_s106
                                 + 0.012098884826996 w_k1_AXP_s107
                                 + 0.0239943519834082 w_k1_AXP_s110
                                 + 0.0834779078983343 w_k1_AXP_s111
                                 + 0.0199850792511395 w_k1_AXP_s112
                                 + 0.0264327487534494 w_k1_AXP_s113
                                 + 0.0476374007540492 w_k1_AXP_s114
                                 + 0.0217619165850206 w_k1_AXP_s115
                                 + 0.0306218026655054 w_k1_AXP_s116
                                 + 0.0808940858858583 w_k1_AXP_s117
                                 + 0.0132923864942211 w_k1_DIS_s000
                                 + 0.0514990096953186 w_k1_DIS_s001
                                 + 0.0371371212094909 w_k1_DIS_s002
                                 + 0.0466567877274346 w_k1_DIS_s003
                                 + 0.0376995362283535 w_k1_DIS_s004
                                 + 0.0430470621421116 w_k1_DIS_s005
                                 + 0.0420928756467143 w_k1_DIS_s006
                                 + 0.0609136579108101 w_k1_DIS_s007
                                 + 0.0199026101492563 w_k1_DIS_s010
                                 + 0.0230031159761988 w_k1_DIS_s011
                                 + 0.0153589682185727 w_k1_DIS_s012
                                 + 0.0186686960622951 w_k1_DIS_s013
                                 + 0.0345048324215675 w_k1_DIS_s014
                                 + 0.0216668838604112 w_k1_DIS_s015
                                 + 0.0374030355221863 w_k1_DIS_s016
                                 + 0.0350676557227613 w_k1_DIS_s017
                                 + 0.0534195161314429 w_k1_DIS_s100
                                 + 0.0159746199059162 w_k1_DIS_s101
                                 + 0.0191749589930122 w_k1_DIS_s102
                                 + 0.0124047712534877 w_k1_DIS_s103
                                 + 0.0140273504925294 w_k1_DIS_s104
                                 + 0.0158330773534963 w_k1_DIS_s105
                                 + 0.0124750188513828 w_k1_DIS_s106
                                 + 0.0112463134601479 w_k1_DIS_s107
                                 + 0.0246760600064928 w_k1_DIS_s110
                                 + 0.0885900298036571 w_k1_DIS_s111
                                 + 0.0189775598779697 w_k1_DIS_s112
                                 + 0.0314895831452949 w_k1_DIS_s113
                                 + 0.0448080594656052 w_k1_DIS_s114
                                 + 0.0216553042556705 w_k1_DIS_s115
                                 + 0.0305328346652864 w_k1_DIS_s116
                                 + 0.0838409918099129 w_k1_DIS_s117
                                 + 0.0125787557722322 w_k1_FDX_s000
                                 + 0.0528044894120481 w_k1_FDX_s001
                                 + 0.0386237908902372 w_k1_FDX_s002
                                 + 0.043101488271884 w_k1_FDX_s003
                                 + 0.0378011881304865 w_k1_FDX_s004
                                 + 0.0435365697464927 w_k1_FDX_s005
                                 + 0.0406272598892475 w_k1_FDX_s006
                                 + 0.0578746297546804 w_k1_FDX_s007
                                 + 0.020157092839152 w_k1_FDX_s010
                                 + 0.0219774426763864 w_k1_FDX_s011
                                 + 0.0149257553584886 w_k1_FDX_s012
                                 + 0.0173010628005821 w_k1_FDX_s013
                                 + 0.0364438784930545 w_k1_FDX_s014
                                 + 0.0230995752671466 w_k1_FDX_s015
                                 + 0.0381421547575384 w_k1_FDX_s016
                                 + 0.0356230623242763 w_k1_FDX_s017
                                 + 0.0554335246547839 w_k1_FDX_s100
                                 + 0.0146843413551434 w_k1_FDX_s101
                                 + 0.0194566637128538 w_k1_FDX_s102
                                 + 0.011379102976396 w_k1_FDX_s103
                                 + 0.0123229147669431 w_k1_FDX_s104
                                 + 0.0160773794298977 w_k1_FDX_s105
                                 + 0.0119070282848323 w_k1_FDX_s106
                                 + 0.0121240303325939 w_k1_FDX_s107
                                 + 0.0238752319145811 w_k1_FDX_s110
                                 + 0.0829898994318774 w_k1_FDX_s111
                                 + 0.0190046163243187 w_k1_FDX_s112
                                 + 0.0300100663023911 w_k1_FDX_s113
                                 + 0.0426323150700112 w_k1_FDX_s114
                                 + 0.0212994643893006 w_k1_FDX_s115
                                 + 0.0303694492881083 w_k1_FDX_s116
                                 + 0.0779770934582407 w_k1_FDX_s117
                                 + 0.0121238156528951 w_k1_IBM_s000
                                 + 0.0534259795987096 w_k1_IBM_s001
                                 + 0.0367102080904821 w_k1_IBM_s002
                                 + 0.0436390509524558 w_k1_IBM_s003
                                 + 0.038222399342339 w_k1_IBM_s004
                                 + 0.045448110498643 w_k1_IBM_s005
                                 + 0.0412159365514086 w_k1_IBM_s006
                                 + 0.0557603197674177 w_k1_IBM_s007
                                 + 0.0217391265883787 w_k1_IBM_s010
                                 + 0.0215761886253368 w_k1_IBM_s011
                                 + 0.0155679175690032 w_k1_IBM_s012
                                 + 0.0174792296504482 w_k1_IBM_s013
                                 + 0.036638302504198 w_k1_IBM_s014
                                 + 0.0227721083162663 w_k1_IBM_s015
                                 + 0.039801269883118 w_k1_IBM_s016
                                 + 0.0358216750536154 w_k1_IBM_s017
                                 + 0.0584854310176165 w_k1_IBM_s100
                                 + 0.0145433594512891 w_k1_IBM_s101
                                 + 0.0198212414198914 w_k1_IBM_s102
                                 + 0.01185970629298 w_k1_IBM_s103
                                 + 0.0122132172347504 w_k1_IBM_s104
                                 + 0.0155739455191519 w_k1_IBM_s105
                                 + 0.0120598376341153 w_k1_IBM_s106
                                 + 0.0130250522738341 w_k1_IBM_s107
                                 + 0.0248493603560068 w_k1_IBM_s110
                                 + 0.0844497228493944 w_k1_IBM_s111
                                 + 0.0184205048736363 w_k1_IBM_s112
                                 + 0.0292692485156886 w_k1_IBM_s113
                                 + 0.0436798764740393 w_k1_IBM_s114
                                 + 0.0223305150849766 w_k1_IBM_s115
                                 + 0.0304428992295121 w_k1_IBM_s116
                                 + 0.0792440253396164 w_k1_IBM_s117
                                 + 0.0140674547507402 w_k1_INTC_s000
                                 + 0.0512102980273753 w_k1_INTC_s001
                                 + 0.0369037907806708 w_k1_INTC_s002
                                 + 0.0432157188449396 w_k1_INTC_s003
                                 + 0.0328045068804294 w_k1_INTC_s004
                                 + 0.0437743954543744 w_k1_INTC_s005
                                 + 0.0418678320380709 w_k1_INTC_s006
                                 + 0.0570586686663554 w_k1_INTC_s007
                                 + 0.0181481013443609 w_k1_INTC_s010
                                 + 0.0262813957289088 w_k1_INTC_s011
                                 + 0.0132959895724632 w_k1_INTC_s012
                                 + 0.0171158813469797 w_k1_INTC_s013
                                 + 0.0341091340440981 w_k1_INTC_s014
                                 + 0.0260619872149051 w_k1_INTC_s015
                                 + 0.0376057109968676 w_k1_INTC_s016
                                 + 0.0357288151680556 w_k1_INTC_s017
                                 + 0.0590021844255613 w_k1_INTC_s100
                                 + 0.0141829357253987 w_k1_INTC_s101
                                 + 0.0191814227012978 w_k1_INTC_s102
                                 + 0.0114450260068042 w_k1_INTC_s103
                                 + 0.0123215947857579 w_k1_INTC_s104
                                 + 0.0160954956465387 w_k1_INTC_s105
                                 + 0.0113160071898492 w_k1_INTC_s106
                                 + 0.012481794674909 w_k1_INTC_s107
                                 + 0.0229044201207729 w_k1_INTC_s110
                                 + 0.0897136415043556 w_k1_INTC_s111
                                 + 0.0217314035987614 w_k1_INTC_s112
                                 + 0.0265276815052771 w_k1_INTC_s113
                                 + 0.0383367633222987 w_k1_INTC_s114
                                 + 0.0205401255276173 w_k1_INTC_s115
                                 + 0.0306114389921539 w_k1_INTC_s116
                                 + 0.068993008762088 w_k1_INTC_s117
                                 + 0.0122158033214574 w_k1_KO_s000
                                 + 0.0547087110260338 w_k1_KO_s001
                                 + 0.0389560966281392 w_k1_KO_s002
                                 + 0.0422701228208563 w_k1_KO_s003
                                 + 0.0395939235432501 w_k1_KO_s004
                                 + 0.0405277182241575 w_k1_KO_s005
                                 + 0.0395246246906555 w_k1_KO_s006
                                 + 0.0659095082954443 w_k1_KO_s007
                                 + 0.0199766465011084 w_k1_KO_s010
                                 + 0.0200933988757724 w_k1_KO_s011
                                 + 0.0154337554134794 w_k1_KO_s012
                                 + 0.0175707201957456 w_k1_KO_s013
                                 + 0.0349749464110541 w_k1_KO_s014
                                 + 0.0230014471307732 w_k1_KO_s015
                                 + 0.0384444610155703 w_k1_KO_s016
                                 + 0.0316689821269072 w_k1_KO_s017
                                 + 0.0598135436695463 w_k1_KO_s100
                                 + 0.015217727436622 w_k1_KO_s101
                                 + 0.0176170349840615 w_k1_KO_s102
                                 + 0.0131511096893905 w_k1_KO_s103
                                 + 0.0120825621558066 w_k1_KO_s104
                                 + 0.0153112171726856 w_k1_KO_s105
                                 + 0.0126752292021971 w_k1_KO_s106
                                 + 0.0124602053833582 w_k1_KO_s107
                                 + 0.0239915266451165 w_k1_KO_s110
                                 + 0.0893349921614017 w_k1_KO_s111
                                 + 0.0196615992499409 w_k1_KO_s112
                                 + 0.0266395712843104 w_k1_KO_s113
                                 + 0.0431030746700991 w_k1_KO_s114
                                 + 0.0202583145649908 w_k1_KO_s115
                                 + 0.0287246324704748 w_k1_KO_s116
                                 + 0.075899638498177 w_k1_KO_s117
                                 + 0.0100909532465565 w_k1_MAT_s000
                                 + 0.0652659636293807 w_k1_MAT_s001
                                 + 0.0323989182227344 w_k1_MAT_s002
                                 + 0.0443451969212392 w_k1_MAT_s003
                                 + 0.0369499241372952 w_k1_MAT_s004
                                 + 0.0368974688870478 w_k1_MAT_s005
                                 + 0.0365901825203348 w_k1_MAT_s006
                                 + 0.0591126860352357 w_k1_MAT_s007
                                 + 0.024307051195721 w_k1_MAT_s010
                                 + 0.0220196721014052 w_k1_MAT_s011
                                 + 0.0172890129111282 w_k1_MAT_s012
                                 + 0.0172046596287342 w_k1_MAT_s013
                                 + 0.0427078964222702 w_k1_MAT_s014
                                 + 0.0185057692426903 w_k1_MAT_s015
                                 + 0.0414058881478426 w_k1_MAT_s016
                                 + 0.0401470141973003 w_k1_MAT_s017
                                 + 0.0447612635694136 w_k1_MAT_s100
                                 + 0.014872831420605 w_k1_MAT_s101
                                 + 0.0217604662628135 w_k1_MAT_s102
                                 + 0.0102929477733406 w_k1_MAT_s103
                                 + 0.00995919673265131 w_k1_MAT_s104
                                 + 0.014945736701444 w_k1_MAT_s105
                                 + 0.0129371411917513 w_k1_MAT_s106
                                 + 0.0141133501670023 w_k1_MAT_s107
                                 + 0.0246755485650273 w_k1_MAT_s110
                                 + 0.0922660366892084 w_k1_MAT_s111
                                 + 0.0217836117021892 w_k1_MAT_s112
                                 + 0.024686292273012 w_k1_MAT_s113
                                 + 0.0528480546414758 w_k1_MAT_s114
                                 + 0.0200558531932516 w_k1_MAT_s115
                                 + 0.0260236583455877 w_k1_MAT_s116
                                 + 0.0761668902207287 w_k1_MAT_s117
                                 + 0.0136582973715695 w_k1_MSFT_s000
                                 + 0.0566926574865479 w_k1_MSFT_s001
                                 + 0.0367872966846644 w_k1_MSFT_s002
                                 + 0.0440480511219773 w_k1_MSFT_s003
                                 + 0.0405870102203346 w_k1_MSFT_s004
                                 + 0.0447877117668953 w_k1_MSFT_s005
                                 + 0.0408244295131605 w_k1_MSFT_s006
                                 + 0.0588468125879737 w_k1_MSFT_s007
                                 + 0.0193888392338191 w_k1_MSFT_s010
                                 + 0.0231796069223562 w_k1_MSFT_s011
                                 + 0.0135987893543348 w_k1_MSFT_s012
                                 + 0.0168612114214467 w_k1_MSFT_s013
                                 + 0.0377122707995591 w_k1_MSFT_s014
                                 + 0.0246986820082947 w_k1_MSFT_s015
                                 + 0.035710881503607 w_k1_MSFT_s016
                                 + 0.0308203664149042 w_k1_MSFT_s017
                                 + 0.0552730492925257 w_k1_MSFT_s100
                                 + 0.0146170159087413 w_k1_MSFT_s101
                                 + 0.0193428018483718 w_k1_MSFT_s102
                                 + 0.0115101306382175 w_k1_MSFT_s103
                                 + 0.0113425328639297 w_k1_MSFT_s104
                                 + 0.0154245605044659 w_k1_MSFT_s105
                                 + 0.0111187385727897 w_k1_MSFT_s106
                                 + 0.0117280218636023 w_k1_MSFT_s107
                                 + 0.0224922435983097 w_k1_MSFT_s110
                                 + 0.0846240177060996 w_k1_MSFT_s111
                                 + 0.0191373591597224 w_k1_MSFT_s112
                                 + 0.0250132297342638 w_k1_MSFT_s113
                                 + 0.0390512868960238 w_k1_MSFT_s114
                                 + 0.0228345261633073 w_k1_MSFT_s115
                                 + 0.0275456285549567 w_k1_MSFT_s116
                                 + 0.0789528682270069 w_k1_MSFT_s117
                                 + 0.0130735654893725 w_k1_PG_s000
                                 + 0.0538811741502167 w_k1_PG_s001
                                 + 0.0379081288768195 w_k1_PG_s002
                                 + 0.0422715728543539 w_k1_PG_s003
                                 + 0.0374617303446265 w_k1_PG_s004
                                 + 0.0410916372274239 w_k1_PG_s005
                                 + 0.0391548554357019 w_k1_PG_s006
                                 + 0.0588257280906367 w_k1_PG_s007
                                 + 0.019834599816735 w_k1_PG_s010
                                 + 0.0205139288849344 w_k1_PG_s011
                                 + 0.0143282002383593 w_k1_PG_s012
                                 + 0.0174491375070434 w_k1_PG_s013
                                 + 0.0335091919337352 w_k1_PG_s014
                                 + 0.0223422654025531 w_k1_PG_s015
                                 + 0.03841037761232 w_k1_PG_s016
                                 + 0.0352725790332009 w_k1_PG_s017
                                 + 0.0560187987236274 w_k1_PG_s100
                                 + 0.01458368727784 w_k1_PG_s101
                                 + 0.0179863991165771 w_k1_PG_s102
                                 + 0.0112882570080925 w_k1_PG_s103
                                 + 0.0127465793894689 w_k1_PG_s104
                                 + 0.0152121479375027 w_k1_PG_s105
                                 + 0.012771017417117 w_k1_PG_s106
                                 + 0.0127647529898792 w_k1_PG_s107
                                 + 0.024353648708172 w_k1_PG_s110
                                 + 0.0809175453719666 w_k1_PG_s111
                                 + 0.0181057123855711 w_k1_PG_s112
                                 + 0.0266982650404166 w_k1_PG_s113
                                 + 0.0423788478977032 w_k1_PG_s114
                                 + 0.0225444230000934 w_k1_PG_s115
                                 + 0.0306837734467752 w_k1_PG_s116
                                 + 0.0799925799963913 w_k1_PG_s117 >= 1.2
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
