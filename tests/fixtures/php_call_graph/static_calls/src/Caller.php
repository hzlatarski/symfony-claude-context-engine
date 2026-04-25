<?php
namespace App;

use App\Service\Helper;

class Caller
{
    public function run(): void
    {
        Helper::staticCall();
        self::helper();
        \App\Service\Helper::other();
    }

    public static function helper(): void {}
}
